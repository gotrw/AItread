"""
MIRA_trader_C – Multi-Strategy Voting Engine.

Combines signals from multiple strategies using a majority-vote approach.
A trade is only opened when at least `min_votes` strategies agree on direction.
"""
from __future__ import annotations

import logging
from typing import List

import pandas as pd

from strategies.base import BaseStrategy, Signal
from strategies.trend_follow import TrendFollowStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.breakout import BreakoutStrategy

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, type[BaseStrategy]] = {
    "trend_follow": TrendFollowStrategy,
    "mean_reversion": MeanReversionStrategy,
    "breakout": BreakoutStrategy,
}


def build_strategy(name: str, params: dict) -> BaseStrategy:
    """Instantiate a strategy by name."""
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy '{name}'. Available: {list(_REGISTRY)}")
    return cls(params)


class MultiStrategy(BaseStrategy):
    """Aggregate signal from multiple child strategies by voting."""

    name = "multi_strategy"

    def __init__(self, params: dict) -> None:
        super().__init__(params)
        ms_cfg = params.get("multi_strategy", {})
        strategy_names: List[str] = ms_cfg.get("strategies", ["trend_follow"])
        self._min_votes: int = ms_cfg.get("min_votes", 2)
        self._strategies: List[BaseStrategy] = [
            build_strategy(n, params.get(n, {})) for n in strategy_names
        ]

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        votes: dict[str, list[Signal]] = {"buy": [], "sell": [], "hold": []}
        for strategy in self._strategies:
            sig = strategy.generate_signal(df)
            votes[sig.action].append(sig)
            logger.debug("Strategy %s → %s (conf=%.2f)", strategy.name, sig.action, sig.confidence)

        buy_count = len(votes["buy"])
        sell_count = len(votes["sell"])

        if buy_count >= self._min_votes:
            avg_conf = sum(s.confidence for s in votes["buy"]) / buy_count
            return Signal(
                action="buy",
                confidence=avg_conf,
                meta={"buy_votes": buy_count, "sell_votes": sell_count},
            )
        if sell_count >= self._min_votes:
            avg_conf = sum(s.confidence for s in votes["sell"]) / sell_count
            return Signal(
                action="sell",
                confidence=avg_conf,
                meta={"buy_votes": buy_count, "sell_votes": sell_count},
            )
        return Signal(action="hold", meta={"buy_votes": buy_count, "sell_votes": sell_count})
