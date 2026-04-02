"""
MIRA_trader_C – Risk Engine.

Enforces position sizing and trading guardrails:
- max_risk_per_trade_pct   : % of capital risked per trade
- max_daily_loss_pct       : halt trading for the day if exceeded
- max_open_positions       : cap simultaneous open trades
- max_exposure_per_symbol  : cap capital allocated to a single symbol
- cooldown_after_loss      : skip N trades after a losing trade
- kill_switch              : hard-stop flag

Usage
-----
    engine = RiskEngine(risk_cfg, initial_capital=1000.0)
    # Before every potential trade:
    approved, reason = engine.approve_trade(symbol, signal, current_price, open_positions)
    if approved:
        qty, sl, tp = engine.compute_sizing(current_price, stop_loss_pct, take_profit_pct)
    # After trade closes:
    engine.record_trade_result(pnl_pct)
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Dict, Tuple

from strategies.base import Signal

logger = logging.getLogger(__name__)


class RiskEngine:
    """Centralised risk manager."""

    def __init__(self, risk_cfg: dict, initial_capital: float) -> None:
        self._cfg = risk_cfg
        self._capital = initial_capital
        self._daily_pnl: float = 0.0
        self._trade_date: date = date.today()
        self._cooldown_remaining: int = 0

    # ── Public helpers ─────────────────────────────────────────

    @property
    def capital(self) -> float:
        return self._capital

    def update_capital(self, new_capital: float) -> None:
        self._capital = new_capital

    # ── Core gate ──────────────────────────────────────────────

    def approve_trade(
        self,
        symbol: str,
        signal: Signal,
        price: float,
        open_positions: Dict[str, dict],
    ) -> Tuple[bool, str]:
        """Return (approved, reason).  reason is non-empty when rejected."""

        self._maybe_reset_daily()

        if self._cfg.get("kill_switch", False):
            return False, "kill_switch is active"

        if not signal.is_actionable():
            return False, "signal is hold"

        # Daily loss guard
        max_daily_loss = self._cfg.get("max_daily_loss_pct", 3.0)
        if self._daily_pnl <= -max_daily_loss:
            return False, f"daily loss limit reached ({self._daily_pnl:.2f}%)"

        # Cooldown
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return False, f"cooldown ({self._cooldown_remaining + 1} trades remaining)"

        # Open position count
        max_pos = self._cfg.get("max_open_positions", 3)
        if len(open_positions) >= max_pos:
            return False, f"max open positions ({max_pos}) reached"

        # Symbol exposure
        max_exposure = self._cfg.get("max_exposure_per_symbol_pct", 30) / 100
        if symbol in open_positions:
            existing = open_positions[symbol].get("value", 0)
            if existing / self._capital >= max_exposure:
                return False, f"max exposure per symbol ({max_exposure*100:.0f}%) reached"

        return True, ""

    def compute_sizing(
        self,
        price: float,
        sl_pct: float | None = None,
        tp_pct: float | None = None,
    ) -> Tuple[float, float, float]:
        """Return (quantity, stop_loss_price, take_profit_price).

        Quantity is calculated so that the risk (distance from entry to SL)
        equals max_risk_per_trade_pct of capital.
        """
        sl_pct = sl_pct or self._cfg.get("stop_loss_pct", 1.5)
        tp_pct = tp_pct or self._cfg.get("take_profit_pct", 3.0)

        risk_pct = self._cfg.get("max_risk_per_trade_pct", 1.0) / 100
        risk_amount = self._capital * risk_pct

        sl_distance = price * sl_pct / 100
        qty = risk_amount / sl_distance if sl_distance > 0 else 0.0

        stop_loss_price = price - sl_distance
        take_profit_price = price + price * tp_pct / 100

        logger.debug(
            "Sizing: qty=%.6f SL=%.4f TP=%.4f (capital=%.2f risk_pct=%.2f%%)",
            qty,
            stop_loss_price,
            take_profit_price,
            self._capital,
            risk_pct * 100,
        )
        return qty, stop_loss_price, take_profit_price

    def record_trade_result(self, pnl_pct: float) -> None:
        """Update running state after a trade closes."""
        self._daily_pnl += pnl_pct
        logger.info("Trade result: pnl_pct=%.2f%% daily_pnl=%.2f%%", pnl_pct, self._daily_pnl)

        if pnl_pct < 0:
            cooldown = self._cfg.get("cooldown_after_loss_trades", 3)
            self._cooldown_remaining = cooldown
            logger.warning(
                "Losing trade. Entering cooldown for %d trades.", cooldown
            )

    # ── Internal ───────────────────────────────────────────────

    def _maybe_reset_daily(self) -> None:
        today = date.today()
        if today != self._trade_date:
            self._trade_date = today
            self._daily_pnl = 0.0
            logger.info("New trading day – daily PnL reset.")
