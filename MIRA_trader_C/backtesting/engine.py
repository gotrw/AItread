"""
MIRA_trader_C – Backtesting Engine.

A realistic bar-by-bar backtesting engine with full futures support.
- Iterates over OHLCV candles one at a time (no look-ahead)
- Applies configurable fee and slippage to every fill
- Supports long AND short positions
- Leverage: PnL is multiplied by leverage, margin is capital/leverage
- Funding rate: charged every N hours (configurable interval)
- Liquidation: force-closes position if price hits the liquidation price
- Tracks open positions, realised PnL, and equity curve

Usage
-----
    from backtesting.engine import BacktestEngine
    from strategies.multi_strategy import MultiStrategy

    strategy = MultiStrategy(cfg["strategy"])
    engine = BacktestEngine(cfg, strategy)
    results = engine.run(df, symbol='BTC/USDT')
    print(results.summary())
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import List, Optional

import pandas as pd

from risk.engine import RiskEngine
from strategies.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Trade record
# ──────────────────────────────────────────────────────────────

@dataclass
class Trade:
    symbol: str
    entry_time: pd.Timestamp
    entry_price: float
    qty: float
    stop_loss: float
    take_profit: float
    side: str = "long"           # 'long' or 'short'
    leverage: int = 1
    liquidation_price: float = 0.0
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0             # absolute USD (after leverage)
    pnl_pct: float = 0.0
    exit_reason: str = ""
    funding_paid: float = 0.0    # total funding cost charged


# ──────────────────────────────────────────────────────────────
# Results container
# ──────────────────────────────────────────────────────────────

@dataclass
class BacktestResults:
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    initial_capital: float = 1000.0

    def summary(self) -> dict:
        if not self.trades:
            return {"error": "No trades executed."}

        closed = [t for t in self.trades if t.exit_price is not None]
        if not closed:
            return {"error": "No closed trades."}

        wins = [t for t in closed if t.pnl > 0]
        losses = [t for t in closed if t.pnl <= 0]
        longs = [t for t in closed if t.side == "long"]
        shorts = [t for t in closed if t.side == "short"]
        liquidations = [t for t in closed if t.exit_reason == "liquidated"]

        total_pnl = sum(t.pnl for t in closed)
        total_funding = sum(t.funding_paid for t in closed)
        win_rate = len(wins) / len(closed) if closed else 0
        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0
        profit_factor = (
            abs(sum(t.pnl for t in wins) / sum(t.pnl for t in losses))
            if losses and sum(t.pnl for t in losses) != 0
            else float("inf")
        )

        equity = self.equity_curve
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max.replace(0, float("nan"))
        max_drawdown = drawdown.min() * 100

        final_capital = self.initial_capital + total_pnl
        total_return_pct = (final_capital - self.initial_capital) / self.initial_capital * 100

        return {
            "total_trades": len(closed),
            "long_trades": len(longs),
            "short_trades": len(shorts),
            "liquidations": len(liquidations),
            "win_rate": f"{win_rate:.1%}",
            "profit_factor": f"{profit_factor:.2f}",
            "total_pnl": f"${total_pnl:.2f}",
            "funding_paid": f"${total_funding:.2f}",
            "total_return_pct": f"{total_return_pct:.2f}%",
            "avg_win": f"${avg_win:.2f}",
            "avg_loss": f"${avg_loss:.2f}",
            "max_drawdown": f"{max_drawdown:.2f}%",
            "final_capital": f"${final_capital:.2f}",
        }


# ──────────────────────────────────────────────────────────────
# Engine
# ──────────────────────────────────────────────────────────────

class BacktestEngine:
    """Bar-by-bar backtesting engine with futures simulation."""

    def __init__(self, cfg: dict, strategy: BaseStrategy) -> None:
        self._cfg = cfg
        self._strategy = strategy
        self._bt_cfg = cfg.get("backtest", {})
        self._risk_cfg = cfg.get("risk", {})
        self._futures_cfg = cfg.get("futures", {})
        self._initial_capital = self._risk_cfg.get("initial_capital", 1000.0)
        self._fee_rate = self._bt_cfg.get("fee_rate", 0.0004)
        self._slippage_pct = self._bt_cfg.get("slippage_pct", 0.05) / 100
        self._simulate_funding = self._bt_cfg.get("simulate_funding", True)
        self._simulate_liquidation = self._bt_cfg.get("simulate_liquidation", True)
        self._leverage: int = self._futures_cfg.get("leverage", 1)
        self._funding_interval_h: int = self._futures_cfg.get("funding_rate_interval_hours", 8)
        self._default_funding_rate: float = self._futures_cfg.get("default_funding_rate", 0.0001)
        self._liquidation_buffer_pct: float = self._futures_cfg.get("liquidation_buffer_pct", 20.0)

    def run(self, df: pd.DataFrame, symbol: str = "BTC/USDT") -> BacktestResults:
        """Run the backtest over the given OHLCV DataFrame."""
        capital = self._initial_capital
        risk_engine = RiskEngine(self._risk_cfg, capital)

        open_trade: Optional[Trade] = None
        trades: List[Trade] = []
        equity: List[float] = []

        # Minimum warmup candles for indicators
        warmup = max(
            self._cfg.get("strategy", {}).get("trend_follow", {}).get("slow_ema", 21),
            self._cfg.get("strategy", {}).get("mean_reversion", {}).get("bb_period", 20),
            self._cfg.get("strategy", {}).get("breakout", {}).get("lookback", 20),
            self._cfg.get("strategy", {}).get("regime", {}).get("lookback", 50),
        ) + 5

        for i in range(warmup, len(df)):
            window = df.iloc[: i + 1]
            row = df.iloc[i]
            ts = df.index[i]
            price = float(row["close"])
            high = float(row["high"])
            low = float(row["low"])

            # ── Manage open position ───────────────────────────
            if open_trade is not None:
                closed = False
                exit_price = price
                exit_reason = ""

                # Funding rate charge (every N hours)
                if self._simulate_funding:
                    funding_pct = self._apply_funding(open_trade, ts, self._default_funding_rate)
                    if funding_pct != 0:
                        funding_cost = abs(funding_pct) * capital / 100
                        open_trade.funding_paid += funding_cost
                        capital -= funding_cost

                # Check liquidation first
                if self._simulate_liquidation and open_trade.liquidation_price > 0:
                    if open_trade.side == "long" and low <= open_trade.liquidation_price:
                        exit_price = open_trade.liquidation_price
                        exit_reason = "liquidated"
                        closed = True
                    elif open_trade.side == "short" and high >= open_trade.liquidation_price:
                        exit_price = open_trade.liquidation_price
                        exit_reason = "liquidated"
                        closed = True

                # Check SL/TP
                if not closed:
                    if open_trade.side == "long":
                        if low <= open_trade.stop_loss:
                            exit_price = open_trade.stop_loss
                            exit_reason = "stop_loss"
                            closed = True
                        elif high >= open_trade.take_profit:
                            exit_price = open_trade.take_profit
                            exit_reason = "take_profit"
                            closed = True
                    else:  # short
                        if high >= open_trade.stop_loss:
                            exit_price = open_trade.stop_loss
                            exit_reason = "stop_loss"
                            closed = True
                        elif low <= open_trade.take_profit:
                            exit_price = open_trade.take_profit
                            exit_reason = "take_profit"
                            closed = True

                if closed:
                    capital, open_trade = self._close_trade(
                        open_trade, exit_price, exit_reason, ts, capital, risk_engine
                    )
                    trades.append(open_trade)
                    open_trade = None

            equity.append(capital)

            # ── Generate signal for new position ───────────────
            if open_trade is not None:
                continue

            signal: Signal = self._strategy.generate_signal(window)
            if not signal.is_entry():
                continue

            side = signal.direction()
            approved, reason = risk_engine.approve_trade(
                symbol, signal, price, {symbol: {"value": 0}} if open_trade is None else {}
            )
            if not approved:
                logger.debug("Trade rejected: %s", reason)
                continue

            # Apply entry slippage
            if side == "long":
                entry_price = price * (1 + self._slippage_pct)
            else:
                entry_price = price * (1 - self._slippage_pct)

            entry_fee = entry_price * self._fee_rate

            qty, sl_price, tp_price = risk_engine.compute_sizing(
                entry_price,
                signal.stop_loss_pct or None,
                signal.take_profit_pct or None,
                side=side,
                leverage=self._leverage,
            )

            # Compute liquidation price (isolated margin)
            margin_rate = 1.0 / max(self._leverage, 1)
            maintenance = 0.005
            if side == "long":
                liq_price = entry_price * (1 - margin_rate + maintenance)
            else:
                liq_price = entry_price * (1 + margin_rate - maintenance)

            cost = (entry_price * qty) / self._leverage + entry_fee * qty
            if cost > capital:
                qty = max(
                    (capital * 0.99 * self._leverage) / (entry_price * (1 + self._fee_rate)),
                    0,
                )

            open_trade = Trade(
                symbol=symbol,
                entry_time=ts,
                entry_price=entry_price,
                qty=qty,
                stop_loss=sl_price,
                take_profit=tp_price,
                side=side,
                leverage=self._leverage,
                liquidation_price=liq_price,
            )
            logger.debug(
                "Opened %s %s @ %.4f qty=%.6f lev=%dx liq=%.4f",
                side, symbol, entry_price, qty, self._leverage, liq_price,
            )

        # Close any remaining open position at last price
        if open_trade is not None:
            final_price = float(df.iloc[-1]["close"])
            capital, open_trade = self._close_trade(
                open_trade, final_price, "end_of_data", df.index[-1], capital, risk_engine
            )
            trades.append(open_trade)

        equity_series = pd.Series(equity, index=df.index[warmup:], name="equity")
        return BacktestResults(
            trades=trades,
            equity_curve=equity_series,
            initial_capital=self._initial_capital,
        )

    # ── Helpers ────────────────────────────────────────────────

    def _close_trade(
        self,
        trade: Trade,
        exit_price: float,
        exit_reason: str,
        ts: pd.Timestamp,
        capital: float,
        risk_engine: RiskEngine,
    ) -> tuple[float, Trade]:
        """Finalise a trade and return updated capital + trade record."""
        # Apply exit slippage (unfavourable direction)
        if trade.side == "long":
            exit_adj = exit_price * (1 - self._slippage_pct)
            pnl_raw = (exit_adj - trade.entry_price) * trade.qty
        else:
            exit_adj = exit_price * (1 + self._slippage_pct)
            pnl_raw = (trade.entry_price - exit_adj) * trade.qty

        # Apply leverage to PnL
        pnl_leveraged = pnl_raw * trade.leverage
        fee = exit_adj * trade.qty * self._fee_rate
        pnl = pnl_leveraged - fee - trade.funding_paid

        trade.exit_time = ts
        trade.exit_price = exit_adj
        trade.pnl = pnl
        trade.pnl_pct = pnl / max(capital, 1) * 100
        trade.exit_reason = exit_reason

        new_capital = capital + pnl
        risk_engine.update_capital(new_capital)
        risk_engine.record_trade_result(trade.pnl_pct)

        logger.debug(
            "Closed %s %s @ %.4f (%s) pnl=%.2f (leveraged×%d)",
            trade.side, trade.symbol, exit_adj, exit_reason, pnl, trade.leverage,
        )
        return new_capital, trade

    def _apply_funding(
        self, trade: Trade, current_ts: pd.Timestamp, funding_rate: float
    ) -> float:
        """Return the funding % to charge this candle (0 if not a funding period).

        Funding is charged every ``funding_interval_h`` hours.
        For longs: negative funding rate = receive; positive = pay.
        For shorts: opposite.
        """
        interval = timedelta(hours=self._funding_interval_h)
        candle_time = current_ts.to_pydatetime()
        # Check if the candle timestamp aligns with a funding interval
        epoch = pd.Timestamp("1970-01-01", tz="UTC").to_pydatetime()
        elapsed = candle_time - epoch if candle_time.tzinfo else candle_time - epoch.replace(tzinfo=None)
        is_funding_candle = int(elapsed.total_seconds() / 3600) % self._funding_interval_h == 0

        if not is_funding_candle:
            return 0.0

        # Longs pay funding when positive; shorts pay when negative
        if trade.side == "long":
            return funding_rate * 100  # as percentage
        return -funding_rate * 100

