"""
MIRA_trader_C – Base Strategy interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd


SignalType = Literal["buy", "sell", "hold"]


@dataclass
class Signal:
    """Trading signal produced by a strategy."""

    action: SignalType = "hold"
    confidence: float = 0.0        # 0.0 – 1.0
    stop_loss_pct: float = 0.0     # suggested SL override (0 = use config default)
    take_profit_pct: float = 0.0   # suggested TP override (0 = use config default)
    meta: dict = field(default_factory=dict)

    def is_actionable(self) -> bool:
        return self.action in ("buy", "sell")


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
