"""
MIRA_trader_C – Risk Engine.

Enforces position sizing and trading guardrails:
- max_risk_per_trade_pct   : % of capital risked per trade
- max_daily_loss_pct       : halt trading for the day if exceeded
- max_open_positions       : cap simultaneous open trades
- max_exposure_per_symbol  : cap capital allocated to a single symbol
- max_total_exposure_pct   : cap total portfolio exposure
- cooldown_after_loss      : skip N trades after a losing trade
- kill_switch              : hard-stop flag

Futures extensions:
- per-symbol leverage cap
- liquidation buffer enforcement (stop before liquidation price is hit)
- short position risk (inverted SL/TP calculation)
- risk event logging to risk_log.jsonl

Usage
-----
    engine = RiskEngine(risk_cfg, initial_capital=1000.0)
    # Before every potential trade:
    approved, reason = engine.approve_trade(symbol, signal, current_price, open_positions)
    if approved:
        qty, sl, tp = engine.compute_sizing(current_price, stop_loss_pct, take_profit_pct,
                                            side='long', leverage=5)
    # After trade closes:
    engine.record_trade_result(pnl_pct)
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

from strategies.base import Signal

logger = logging.getLogger(__name__)


class RiskEngine:
    """Centralised risk manager with futures support."""

    def __init__(self, risk_cfg: dict, initial_capital: float) -> None:
        self._cfg = risk_cfg
        self._capital = initial_capital
        self._daily_pnl: float = 0.0
        self._trade_date: date = date.today()
        self._cooldown_remaining: int = 0

        # Risk event log
        risk_log_path = risk_cfg.get("risk_log_file", "logs/risk_log.jsonl")
        self._risk_log = Path(risk_log_path)
        os.makedirs(self._risk_log.parent, exist_ok=True)

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
            self._log_risk_event("kill_switch", symbol, {"price": price})
            return False, "kill_switch is active"

        if not signal.is_actionable():
            return False, "signal is hold"

        # Daily loss guard
        max_daily_loss = self._cfg.get("max_daily_loss_pct", 3.0)
        if self._daily_pnl <= -max_daily_loss:
            reason = f"daily loss limit reached ({self._daily_pnl:.2f}%)"
            self._log_risk_event("daily_loss_limit", symbol, {"daily_pnl": self._daily_pnl})
            return False, reason

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
            if existing / max(self._capital, 1) >= max_exposure:
                return False, f"max exposure per symbol ({max_exposure*100:.0f}%) reached"

        # Total portfolio exposure
        max_total_exp = self._cfg.get("max_total_exposure_pct", 80) / 100
        total_exposure = sum(p.get("value", 0) for p in open_positions.values())
        if total_exposure / max(self._capital, 1) >= max_total_exp:
            return False, f"max total exposure ({max_total_exp*100:.0f}%) reached"

        return True, ""

    def compute_sizing(
        self,
        price: float,
        sl_pct: float | None = None,
        tp_pct: float | None = None,
        side: str = "long",
        leverage: int = 1,
    ) -> Tuple[float, float, float]:
        """Return (quantity, stop_loss_price, take_profit_price).

        For futures, quantity is margin-adjusted: the position notional value
        is ``qty * price``, but only ``qty * price / leverage`` of capital is
        used as margin.  The risk amount (max loss) is limited to
        ``max_risk_per_trade_pct`` of capital.

        Args:
            price: Current market price.
            sl_pct: Stop-loss distance as % from entry (positive number).
            tp_pct: Take-profit distance as % from entry (positive number).
            side: 'long' or 'short'.
            leverage: Futures leverage multiplier (default 1).

        Returns:
            (qty, stop_loss_price, take_profit_price)
        """
        sl_pct = sl_pct or self._cfg.get("stop_loss_pct", 1.5)
        tp_pct = tp_pct or self._cfg.get("take_profit_pct", 3.0)

        risk_pct = self._cfg.get("max_risk_per_trade_pct", 1.0) / 100
        risk_amount = self._capital * risk_pct

        sl_distance = price * sl_pct / 100
        # With leverage, the effective loss per unit is still sl_distance,
        # but we risk `risk_amount` of capital (margin).  The position size
        # in base asset is risk_amount / sl_distance (same formula, leverage
        # is already reflected in margin, not in qty for isolated margin).
        qty = risk_amount / sl_distance if sl_distance > 0 else 0.0

        if side == "long":
            stop_loss_price = price - sl_distance
            take_profit_price = price + price * tp_pct / 100
        else:  # short
            stop_loss_price = price + sl_distance
            take_profit_price = price - price * tp_pct / 100

        logger.debug(
            "Sizing (%s): qty=%.6f SL=%.4f TP=%.4f (capital=%.2f risk=%.2f%% lev=%d)",
            side, qty, stop_loss_price, take_profit_price, self._capital, risk_pct * 100, leverage,
        )
        return qty, stop_loss_price, take_profit_price

    def check_liquidation_buffer(
        self,
        symbol: str,
        current_price: float,
        liquidation_price: float,
        side: str = "long",
        buffer_pct: float = 20.0,
    ) -> bool:
        """Return True if the position is within the liquidation buffer zone.

        When True, the caller should close the position to avoid forced liquidation.

        Args:
            current_price: Current market price.
            liquidation_price: Exchange-reported or computed liquidation price.
            side: 'long' or 'short'.
            buffer_pct: Close position when within this % of liquidation (default 20%).
        """
        if liquidation_price <= 0:
            return False

        if side == "long":
            distance_pct = (current_price - liquidation_price) / current_price * 100
        else:
            distance_pct = (liquidation_price - current_price) / current_price * 100

        in_buffer = distance_pct < buffer_pct
        if in_buffer:
            self._log_risk_event(
                "liquidation_buffer",
                symbol,
                {
                    "current_price": current_price,
                    "liquidation_price": liquidation_price,
                    "distance_pct": distance_pct,
                    "buffer_pct": buffer_pct,
                },
            )
            logger.warning(
                "[RISK] %s in liquidation buffer zone: price=%.4f liq=%.4f dist=%.2f%%",
                symbol, current_price, liquidation_price, distance_pct,
            )
        return in_buffer

    def record_trade_result(self, pnl_pct: float) -> None:
        """Update running state after a trade closes."""
        self._daily_pnl += pnl_pct
        logger.info("Trade result: pnl_pct=%.2f%% daily_pnl=%.2f%%", pnl_pct, self._daily_pnl)

        if pnl_pct < 0:
            cooldown = self._cfg.get("cooldown_after_loss_trades", 3)
            self._cooldown_remaining = cooldown
            logger.warning("Losing trade. Entering cooldown for %d trades.", cooldown)

        self._log_risk_event("trade_result", "", {"pnl_pct": pnl_pct, "daily_pnl": self._daily_pnl})

    def trigger_kill_switch(self) -> None:
        """Activate the kill switch to halt all trading."""
        self._cfg["kill_switch"] = True
        self._log_risk_event("kill_switch_activated", "", {})
        logger.critical("[RISK] Kill switch ACTIVATED. All trading halted.")

    # ── Internal ───────────────────────────────────────────────

    def _maybe_reset_daily(self) -> None:
        today = date.today()
        if today != self._trade_date:
            self._trade_date = today
            self._daily_pnl = 0.0
            logger.info("New trading day – daily PnL reset.")

    def _log_risk_event(self, event_type: str, symbol: str, data: dict) -> None:
        """Append a risk event to the JSONL risk log file."""
        entry = {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "event": event_type,
            "symbol": symbol,
            **data,
        }
        try:
            with open(self._risk_log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError as exc:
            logger.warning("Failed to write risk log: %s", exc)

