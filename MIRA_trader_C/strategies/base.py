"""
MIRA_trader_C – Base Strategy interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd


# Futures-aware signal types: long/short for opening, close_long/close_short for closing
SignalType = Literal["long", "short", "close_long", "close_short", "hold",
                     "buy", "sell"]  # buy/sell kept for backward compatibility


@dataclass
class Signal:
    """Trading signal produced by a strategy."""

    action: SignalType = "hold"
    confidence: float = 0.0        # 0.0 – 1.0
    stop_loss_pct: float = 0.0     # suggested SL override (0 = use config default)
    take_profit_pct: float = 0.0   # suggested TP override (0 = use config default)
    strategy_name: str = ""        # originating strategy name
    regime: str = ""               # detected market regime at signal time
    meta: dict = field(default_factory=dict)

    def is_actionable(self) -> bool:
        return self.action in ("long", "short", "buy", "sell")

    def is_entry(self) -> bool:
        """True for signals that open a new position."""
        return self.action in ("long", "short", "buy")

    def is_exit(self) -> bool:
        """True for signals that close an existing position."""
        return self.action in ("close_long", "close_short", "sell")

    def direction(self) -> str:
        """Return 'long', 'short', or 'flat'."""
        if self.action in ("long", "buy"):
            return "long"
        if self.action in ("short", "sell"):
            return "short"
        if self.action in ("close_long", "close_short"):
            return "flat"
        return "flat"


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    name: str = "base"

    def __init__(self, params: dict) -> None:
        self.params = params

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """Analyse the most recent OHLCV DataFrame and return a Signal.

        The DataFrame is assumed to have enough historical candles loaded.
        The signal should be based only on data up to and including df.iloc[-1].
        """
