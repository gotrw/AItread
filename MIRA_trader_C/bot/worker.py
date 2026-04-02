"""
MIRA_trader_C – Per-Symbol Bot Worker.

Each BotWorker manages the full trading lifecycle for one symbol:
  1. Fetch OHLCV data
  2. Generate signal via per-symbol strategy
  3. Check risk engine approval
  4. Execute order via broker
  5. Monitor open position (SL/TP, liquidation buffer, funding)
  6. Close position and record trade result

Workers run concurrently as asyncio tasks, each sleeping until the next
candle close for their configured timeframe.

Usage (from main.py orchestrator)
----------------------------------
    workers = [
        BotWorker(symbol, sym_cfg, data_provider, broker, risk_engine, metrics, notifier)
        for symbol, sym_cfg in symbol_configs.items()
    ]
    await asyncio.gather(*[w.run() for w in workers])
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from config import load_symbol_config
from data.base import DataProvider
from execution.base import BrokerBase, Order
from monitoring.metrics import MetricsTracker, TradeRecord
from risk.engine import RiskEngine
from strategies.base import BaseStrategy, Signal
from utils.helpers import now_utc

logger = logging.getLogger(__name__)


def _timeframe_to_seconds(tf: str) -> int:
    units = {"m": 60, "h": 3600, "d": 86400}
    if tf and tf[-1] in units:
        return int(tf[:-1]) * units[tf[-1]]
    return 3600


class BotWorker:
    """Autonomous trading worker for a single symbol.

    Parameters
    ----------
    symbol:
        Trading pair, e.g. 'BTC/USDT'.
    global_cfg:
        Global config dict (will be merged with per-symbol overrides).
    data_provider:
        Shared data provider instance.
    broker:
        Broker instance (PaperFuturesBroker or FuturesBroker).
    risk_engine:
        Shared central risk engine.
    metrics:
        Shared metrics tracker.
    notifier:
        Optional TelegramNotifier (may be None).
    strategy:
        Optional pre-built strategy.  If None, built from per-symbol config.
    """

    def __init__(
        self,
        symbol: str,
        global_cfg: dict,
        data_provider: DataProvider,
        broker: BrokerBase,
        risk_engine: RiskEngine,
        metrics: MetricsTracker,
        notifier: Optional[Any] = None,
        strategy: Optional[BaseStrategy] = None,
    ) -> None:
        self.symbol = symbol
        self._global_cfg = global_cfg
        self._cfg = load_symbol_config(symbol, global_cfg)
        self._provider = data_provider
        self._broker = broker
        self._risk_engine = risk_engine
        self._metrics = metrics
        self._notifier = notifier

        # Build per-symbol strategy if not provided
        if strategy is not None:
            self._strategy = strategy
        else:
            self._strategy = self._build_strategy()

        data_cfg = self._cfg.get("data", {})
        self._timeframe: str = data_cfg.get("timeframe", "1h")
        self._lookback: int = data_cfg.get("lookback_candles", 500)
        self._futures_cfg = self._cfg.get("futures", {})
        self._risk_cfg = self._cfg.get("risk", {})
        self._leverage: int = self._futures_cfg.get("leverage", 1)
        self._liquidation_buffer_pct: float = self._cfg.get("futures", {}).get(
            "liquidation_buffer_pct",
            self._global_cfg.get("futures", {}).get("liquidation_buffer_pct", 20.0),
        )

        self._open_position: Optional[Dict[str, Any]] = None  # current open position
        self._tf_seconds: int = _timeframe_to_seconds(self._timeframe)
        self._running: bool = False

        # Trade journal for the learning system
        journal_path = self._cfg.get("monitoring", {}).get(
            "trade_journal_file", "logs/trade_journal.jsonl"
        )
        self._journal_path = Path(journal_path)
        os.makedirs(self._journal_path.parent, exist_ok=True)

    # ── Main loop ──────────────────────────────────────────────

    async def run(self) -> None:
        """Run the trading loop for this symbol until cancelled."""
        self._running = True
        logger.info("[%s] BotWorker started (tf=%s lev=%dx)", self.symbol, self._timeframe, self._leverage)

        # Set leverage and margin type for futures brokers
        await self._configure_futures()

        while self._running:
            try:
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("[%s] Tick error: %s", self.symbol, exc, exc_info=True)
                if self._notifier:
                    self._notifier.send_error(str(exc), self.symbol)

            await asyncio.sleep(self._tf_seconds)

        logger.info("[%s] BotWorker stopped.", self.symbol)

    def stop(self) -> None:
        """Signal the worker to stop after the current tick."""
        self._running = False

    # ── Tick logic ─────────────────────────────────────────────

    async def _tick(self) -> None:
        """One iteration of the trading loop."""
        df = await asyncio.get_event_loop().run_in_executor(
            None, self._provider.fetch_ohlcv, self.symbol, self._timeframe, self._lookback
        )
        if df.empty or len(df) < 30:
            logger.warning("[%s] Insufficient data, skipping tick.", self.symbol)
            return

        price = float(df["close"].iloc[-1])

        # Monitor and manage open position
        if self._open_position is not None:
            await self._manage_position(price, df)

        # If no position, look for a new entry
        if self._open_position is None:
            signal: Signal = await asyncio.get_event_loop().run_in_executor(
                None, self._strategy.generate_signal, df
            )
            logger.info("[%s] price=%.4f signal=%s conf=%.2f", self.symbol, price, signal.action, signal.confidence)

            if signal.is_entry():
                await self._try_enter(signal, price)

    # ── Position management ────────────────────────────────────

    async def _try_enter(self, signal: Signal, price: float) -> None:
        """Attempt to open a new position."""
        open_positions = {self.symbol: self._open_position} if self._open_position else {}
        # Aggregate across all symbols for portfolio-level checks
        approved, reason = self._risk_engine.approve_trade(
            self.symbol, signal, price, open_positions
        )
        if not approved:
            logger.debug("[%s] Trade rejected: %s", self.symbol, reason)
            return

        side = signal.direction()  # 'long' or 'short'
        qty, sl_price, tp_price = self._risk_engine.compute_sizing(
            price,
            signal.stop_loss_pct or None,
            signal.take_profit_pct or None,
            side=side,
            leverage=self._leverage,
        )

        order = await asyncio.get_event_loop().run_in_executor(
            None, self._broker.place_market_order, self.symbol, "buy" if side == "long" else "sell", qty
        )

        # For paper futures brokers, fill the order
        fill_price = price
        if hasattr(self._broker, "fill_paper_order"):
            self._broker.fill_paper_order(order, price, leverage=self._leverage)

        self._open_position = {
            "trade_id": str(uuid.uuid4()),
            "side": side,
            "entry_price": fill_price,
            "qty": qty,
            "sl": sl_price,
            "tp": tp_price,
            "entry_time": now_utc().isoformat(),
            "value": fill_price * qty,
            "leverage": self._leverage,
            "sl_order_id": "",
            "tp_order_id": "",
            # Signal context — used by the learning system
            "strategy": getattr(self._strategy, "name", ""),
            "regime": getattr(signal, "regime", ""),
            "confidence": signal.confidence,
            "timeframe": self._timeframe,
        }

        # For live futures brokers, place exchange-side SL/TP orders
        if hasattr(self._broker, "place_bracket_orders"):
            bracket = await asyncio.get_event_loop().run_in_executor(
                None,
                self._broker.place_bracket_orders,
                self.symbol,
                side,
                qty,
                sl_price,
                tp_price,
            )
            self._open_position["sl_order_id"] = bracket.get("sl_order_id", "")
            self._open_position["tp_order_id"] = bracket.get("tp_order_id", "")
        logger.info(
            "[%s] OPENED %s qty=%.6f @ %.4f SL=%.4f TP=%.4f lev=%dx",
            self.symbol, side.upper(), qty, fill_price, sl_price, tp_price, self._leverage,
        )
        if self._notifier:
            self._notifier.send_order(self.symbol, side, qty, fill_price)

    async def _manage_position(self, price: float, df: Any) -> None:
        """Check SL/TP, liquidation buffer, and close position if needed."""
        pos = self._open_position
        side = pos["side"]
        sl = pos["sl"]
        tp = pos["tp"]
        qty = pos["qty"]
        entry = pos["entry_price"]
        leverage = pos["leverage"]

        close_reason: Optional[str] = None

        # Check stop-loss and take-profit
        if side == "long":
            if price <= sl:
                close_reason = "stop_loss"
            elif price >= tp:
                close_reason = "take_profit"
        else:  # short
            if price >= sl:
                close_reason = "stop_loss"
            elif price <= tp:
                close_reason = "take_profit"

        # Check liquidation buffer (paper: compute synthetic liquidation price)
        if close_reason is None and hasattr(self._broker, "is_liquidated"):
            if self._broker.is_liquidated(self.symbol, price):
                close_reason = "liquidated"
        elif close_reason is None:
            # Estimate liquidation price for live broker check
            margin_rate = 1.0 / max(leverage, 1)
            if side == "long":
                liq_est = entry * (1 - margin_rate + 0.005)
            else:
                liq_est = entry * (1 + margin_rate - 0.005)
            if self._risk_engine.check_liquidation_buffer(
                self.symbol, price, liq_est, side, self._liquidation_buffer_pct
            ):
                close_reason = "liquidation_buffer"

        if close_reason is not None:
            await self._close_position(price, close_reason)

    async def _close_position(self, price: float, reason: str) -> None:
        """Close the open position and record results."""
        pos = self._open_position
        if pos is None:
            return

        side = pos["side"]
        qty = pos["qty"]
        entry = pos["entry_price"]
        leverage = pos["leverage"]
        capital = self._risk_engine.capital

        # Compute PnL (leveraged)
        if side == "long":
            pnl = (price - entry) * qty * leverage
        else:
            pnl = (entry - price) * qty * leverage

        pnl_pct = pnl / max(capital, 1) * 100

        # Place closing order
        close_side = "sell" if side == "long" else "buy"

        # For live futures brokers, cancel exchange-side bracket orders first
        # to avoid orphan reduceOnly orders conflicting with the market close
        if hasattr(self._broker, "cancel_bracket_orders"):
            sl_oid = pos.get("sl_order_id", "")
            tp_oid = pos.get("tp_order_id", "")
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._broker.cancel_bracket_orders,
                self.symbol,
                sl_oid,
                tp_oid,
            )

        order = await asyncio.get_event_loop().run_in_executor(
            None, self._broker.place_market_order, self.symbol, close_side, qty
        )
        if hasattr(self._broker, "fill_paper_order"):
            self._broker.fill_paper_order(order, price, leverage=1)
        if hasattr(self._broker, "close_position"):
            self._broker.close_position(self.symbol)

        self._risk_engine.record_trade_result(pnl_pct)
        new_capital = capital + pnl
        self._risk_engine.update_capital(new_capital)

        self._metrics.record_trade(TradeRecord(
            symbol=self.symbol,
            side=side,
            entry_price=entry,
            exit_price=price,
            qty=qty,
            pnl=pnl,
            pnl_pct=pnl_pct,
            timestamp=now_utc().isoformat(),
            exit_reason=reason,
        ))

        self._write_journal(pos, price, pnl_pct, reason)

        logger.info(
            "[%s] CLOSED %s @ %.4f pnl=%.2f (%.2f%%) reason=%s",
            self.symbol, side.upper(), price, pnl, pnl_pct, reason,
        )
        if self._notifier:
            self._notifier.send_order(self.symbol, close_side, qty, price, pnl=pnl, exit_reason=reason)

        self._open_position = None

    # ── Setup ──────────────────────────────────────────────────

    async def _configure_futures(self) -> None:
        """Set leverage and margin type on the exchange (if supported)."""
        if hasattr(self._broker, "set_leverage"):
            margin_type = self._futures_cfg.get("margin_type", "isolated")
            await asyncio.get_event_loop().run_in_executor(
                None, self._broker.set_leverage, self.symbol, self._leverage
            )
            await asyncio.get_event_loop().run_in_executor(
                None, self._broker.set_margin_type, self.symbol, margin_type
            )

    def _build_strategy(self) -> BaseStrategy:
        """Build a strategy from the per-symbol config, applying any learned weights."""
        from strategies.multi_strategy import MultiStrategy, build_strategy
        strategy_cfg = self._cfg.get("strategy", {})
        active = strategy_cfg.get("active", "multi_strategy")
        if active == "multi_strategy":
            return MultiStrategy(strategy_cfg)
        return build_strategy(active, strategy_cfg.get(active, {}))

    def _write_journal(
        self,
        pos: Dict[str, Any],
        exit_price: float,
        pnl_pct: float,
        exit_reason: str,
    ) -> None:
        """Append a completed trade record to the trade journal."""
        record = {
            "trade_id": pos.get("trade_id", ""),
            "symbol": self.symbol,
            "side": pos.get("side", ""),
            "strategy": pos.get("strategy", ""),
            "regime": pos.get("regime", ""),
            "confidence": pos.get("confidence", 0.0),
            "timeframe": pos.get("timeframe", self._timeframe),
            "entry_price": pos.get("entry_price", 0.0),
            "exit_price": exit_price,
            "qty": pos.get("qty", 0.0),
            "pnl_pct": pnl_pct,
            "win": pnl_pct > 0,
            "entry_time": pos.get("entry_time", ""),
            "exit_time": now_utc().isoformat(),
            "exit_reason": exit_reason,
        }
        try:
            with open(self._journal_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError as exc:
            logger.warning("[%s] Failed to write trade journal: %s", self.symbol, exc)

    @property
    def status(self) -> dict:
        """Return a status dict for dashboard/monitoring."""
        return {
            "symbol": self.symbol,
            "timeframe": self._timeframe,
            "leverage": self._leverage,
            "running": self._running,
            "open_position": self._open_position,
        }
