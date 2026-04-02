"""
MIRA_trader_C – Backtesting Engine.

A simple, realistic bar-by-bar backtesting engine.
- Iterates over OHLCV candles one at a time (no look-ahead)
- Applies configurable fee and slippage to every fill
- Tracks open positions, realised PnL, and equity curve
- Enforces stop-loss and take-profit on each candle

Usage
-----
    from backtesting.engine import BacktestEngine
    from strategies.multi_strategy import MultiStrategy

    strategy = MultiStrategy(cfg["strategy"])
    engine = BacktestEngine(cfg, strategy)
    results = engine.run(df)
    print(results.summary())
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

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
    side: str = "long"
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    pnl: float = 0.0          # absolute USD
    pnl_pct: float = 0.0
    exit_reason: str = ""


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

        total_pnl = sum(t.pnl for t in closed)
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
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100

        final_capital = self.initial_capital + total_pnl
        total_return_pct = (final_capital - self.initial_capital) / self.initial_capital * 100

        return {
            "total_trades": len(closed),
            "win_rate": f"{win_rate:.1%}",
            "profit_factor": f"{profit_factor:.2f}",
            "total_pnl": f"${total_pnl:.2f}",
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
    """Bar-by-bar backtesting engine."""

    def __init__(self, cfg: dict, strategy: BaseStrategy) -> None:
        self._cfg = cfg
        self._strategy = strategy
        self._bt_cfg = cfg.get("backtest", {})
        self._risk_cfg = cfg.get("risk", {})
        self._initial_capital = self._risk_cfg.get("initial_capital", 1000.0)
        self._fee_rate = self._bt_cfg.get("fee_rate", 0.001)
        self._slippage_pct = self._bt_cfg.get("slippage_pct", 0.05) / 100

    def run(self, df: pd.DataFrame, symbol: str = "BTC/USDT") -> BacktestResults:
        """Run the backtest over the given OHLCV DataFrame."""
        capital = self._initial_capital
        risk_engine = RiskEngine(self._risk_cfg, capital)

        open_positions: dict[str, dict] = {}
        trades: List[Trade] = []
        equity: List[float] = []

        # Minimum warmup candles for indicators
        warmup = max(
            self._cfg.get("strategy", {}).get("trend_follow", {}).get("slow_ema", 21),
            self._cfg.get("strategy", {}).get("mean_reversion", {}).get("bb_period", 20),
            self._cfg.get("strategy", {}).get("breakout", {}).get("lookback", 20),
        ) + 5

        for i in range(warmup, len(df)):
            window = df.iloc[: i + 1]
            row = df.iloc[i]
            ts = df.index[i]
            price = float(row["close"])

            # ── Check if open position should be closed ────────
            if symbol in open_positions:
                pos = open_positions[symbol]
                trade: Trade = pos["trade"]
                high = float(row["high"])
                low = float(row["low"])

                closed = False
                exit_reason = ""
                exit_price = price

                if low <= trade.stop_loss:
                    exit_price = trade.stop_loss
                    exit_reason = "stop_loss"
                    closed = True
                elif high >= trade.take_profit:
                    exit_price = trade.take_profit
                    exit_reason = "take_profit"
                    closed = True

                if closed:
                    exit_price_adj = exit_price * (1 - self._slippage_pct)
                    fee = exit_price_adj * trade.qty * self._fee_rate
                    pnl = (exit_price_adj - trade.entry_price) * trade.qty - fee

                    trade.exit_time = ts
                    trade.exit_price = exit_price_adj
                    trade.pnl = pnl
                    trade.pnl_pct = pnl / capital * 100
                    trade.exit_reason = exit_reason

                    capital += pnl
                    risk_engine.update_capital(capital)
                    risk_engine.record_trade_result(trade.pnl_pct)
                    del open_positions[symbol]
                    trades.append(trade)
                    logger.debug("Closed %s at %.4f (%s) pnl=%.2f", symbol, exit_price_adj, exit_reason, pnl)

            equity.append(capital)

            # ── Generate signal ────────────────────────────────
            if symbol in open_positions:
                continue

            signal: Signal = self._strategy.generate_signal(window)
            if signal.action != "buy":
                continue

            approved, reason = risk_engine.approve_trade(
                symbol, signal, price, open_positions
            )
            if not approved:
                logger.debug("Trade rejected: %s", reason)
                continue

            # Apply entry slippage (buy at slightly higher price)
            entry_price = price * (1 + self._slippage_pct)
            entry_fee = entry_price * self._fee_rate

            qty, sl_price, tp_price = risk_engine.compute_sizing(
                entry_price,
                signal.stop_loss_pct or None,
                signal.take_profit_pct or None,
            )

            cost = entry_price * qty + entry_fee * qty
            if cost > capital:
                qty = (capital * 0.99) / (entry_price * (1 + self._fee_rate))

            trade = Trade(
                symbol=symbol,
                entry_time=ts,
                entry_price=entry_price,
                qty=qty,
                stop_loss=sl_price,
                take_profit=tp_price,
            )
            open_positions[symbol] = {"trade": trade, "value": cost}
            logger.debug("Opened %s at %.4f qty=%.6f", symbol, entry_price, qty)

        # Close any remaining open positions at last price
        for sym, pos in open_positions.items():
            trade = pos["trade"]
            final_price = float(df.iloc[-1]["close"])
            fee = final_price * trade.qty * self._fee_rate
            pnl = (final_price - trade.entry_price) * trade.qty - fee
            trade.exit_price = final_price
            trade.exit_time = df.index[-1]
            trade.pnl = pnl
            trade.pnl_pct = pnl / capital * 100
            trade.exit_reason = "end_of_data"
            capital += pnl
            trades.append(trade)

        equity_series = pd.Series(equity, index=df.index[warmup:], name="equity")
        results = BacktestResults(
            trades=trades,
            equity_curve=equity_series,
            initial_capital=self._initial_capital,
        )
        return results
