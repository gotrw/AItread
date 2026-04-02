"""
MIRA_trader_C – Metrics Tracker.

Tracks key performance indicators (signals, trades, PnL, drawdown) in memory
and persists them to a JSON file for external dashboards or alerts.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    pnl_pct: float
    timestamp: str
    exit_reason: str = ""


@dataclass
class Metrics:
    """Running statistics for the live/paper session."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    peak_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    current_drawdown_pct: float = 0.0
    trade_history: List[TradeRecord] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.winning_trades / self.total_trades if self.total_trades else 0.0

    @property
    def profit_factor(self) -> float:
        gross_win = sum(t.pnl for t in self.trade_history if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trade_history if t.pnl <= 0))
        return gross_win / gross_loss if gross_loss else float("inf")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["win_rate"] = f"{self.win_rate:.1%}"
        d["profit_factor"] = f"{self.profit_factor:.2f}"
        return d


class MetricsTracker:
    """Records trades and computes running performance metrics."""

    def __init__(self, metrics_file: str = "logs/metrics.json") -> None:
        self._file = Path(metrics_file)
        os.makedirs(self._file.parent, exist_ok=True)
        self._metrics = Metrics()

    def record_trade(self, record: TradeRecord) -> None:
        self._metrics.trade_history.append(record)
        self._metrics.total_trades += 1
        self._metrics.total_pnl += record.pnl

        if record.pnl > 0:
            self._metrics.winning_trades += 1
        else:
            self._metrics.losing_trades += 1

        # Update drawdown
        if self._metrics.total_pnl > self._metrics.peak_pnl:
            self._metrics.peak_pnl = self._metrics.total_pnl

        if self._metrics.peak_pnl > 0:
            drawdown_pct = (
                (self._metrics.peak_pnl - self._metrics.total_pnl)
                / self._metrics.peak_pnl
                * 100
            )
            self._metrics.current_drawdown_pct = drawdown_pct
            if drawdown_pct > self._metrics.max_drawdown_pct:
                self._metrics.max_drawdown_pct = drawdown_pct

        logger.info(
            "Trade recorded: %s %s pnl=%.2f win_rate=%s",
            record.symbol,
            record.side,
            record.pnl,
            f"{self._metrics.win_rate:.1%}",
        )
        self._persist()

    def get_metrics(self) -> Metrics:
        return self._metrics

    def _persist(self) -> None:
        try:
            with open(self._file, "w", encoding="utf-8") as fh:
                json.dump(self._metrics.to_dict(), fh, indent=2, default=str)
        except OSError as exc:
            logger.warning("Failed to write metrics file: %s", exc)
