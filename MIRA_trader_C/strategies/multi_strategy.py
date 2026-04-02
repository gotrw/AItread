"""
MIRA_trader_C – Multi-Strategy Voting Engine.

Combines signals from multiple strategies using a majority-vote approach.
A trade is only opened when at least `min_votes` strategies agree on direction.

When `regime_aware=True` the engine uses RegimeDetector to identify the
current market regime (trending / ranging / volatile / choppy) and gives
preference to strategies that are best suited for that regime.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from strategies.base import BaseStrategy, Signal
from strategies.trend_follow import TrendFollowStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.breakout import BreakoutStrategy
from strategies.volatility import VolatilityStrategy

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, type[BaseStrategy]] = {
    "trend_follow": TrendFollowStrategy,
    "mean_reversion": MeanReversionStrategy,
    "breakout": BreakoutStrategy,
    "volatility": VolatilityStrategy,
}

# Regime → preferred strategies (ordered by suitability)
_REGIME_PREFERENCE: dict[str, list[str]] = {
    "trending": ["trend_follow", "breakout"],
    "ranging": ["mean_reversion"],
    "volatile": ["volatility", "breakout"],
    "choppy": ["volatility", "mean_reversion"],
}


def build_strategy(name: str, params: dict) -> BaseStrategy:
    """Instantiate a strategy by name."""
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy '{name}'. Available: {list(_REGISTRY)}")
    return cls(params)


class MultiStrategy(BaseStrategy):
    """Aggregate signal from multiple child strategies by voting.

    When regime_aware is True, strategies that match the current regime
    cast double votes, giving a natural preference to the most suitable
    strategy without hard-disabling the others.
    """

    name = "multi_strategy"

    def __init__(self, params: dict) -> None:
        super().__init__(params)
        ms_cfg = params.get("multi_strategy", {})
        strategy_names: List[str] = ms_cfg.get("strategies", ["trend_follow"])
        self._min_votes: int = ms_cfg.get("min_votes", 2)
        self._regime_aware: bool = ms_cfg.get("regime_aware", False)
        self._strategies: List[BaseStrategy] = [
            build_strategy(n, params.get(n, {})) for n in strategy_names
        ]
        # Load learned weights from the learning system (if available)
        self._learned_weights: Dict[str, float] = self._load_learned_weights()
        self._regime_detector = None
        if self._regime_aware:
            from strategies.regime import RegimeDetector
            self._regime_detector = RegimeDetector(params.get("regime", {}))

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        # Detect regime if enabled
        regime = ""
        if self._regime_detector is not None:
            regime = self._regime_detector.detect(df)

        preferred: set[str] = set()
        if regime and regime in _REGIME_PREFERENCE:
            preferred = set(_REGIME_PREFERENCE[regime])

        votes: Dict[str, List[Signal]] = {"long": [], "short": [], "hold": []}
        for strategy in self._strategies:
            sig = strategy.generate_signal(df)
            # Regime-preferred strategies count double
            regime_bonus = 2 if (self._regime_aware and strategy.name in preferred) else 1
            # Apply learned weight (clamped, rounded to nearest int vote count)
            learned_w = self._learned_weights.get(strategy.name, 1.0)
            vote_weight = max(1, round(regime_bonus * learned_w))
            bucket = sig.action if sig.action in ("long", "short", "hold") else "hold"
            for _ in range(vote_weight):
                votes[bucket].append(sig)
            logger.debug(
                "Strategy %s → %s (conf=%.2f, regime_bonus=%d, learned_w=%.2f, votes=%d)",
                strategy.name, sig.action, sig.confidence, regime_bonus, learned_w, vote_weight,
            )

        long_count = len(votes["long"])
        short_count = len(votes["short"])

        if long_count >= self._min_votes:
            avg_conf = sum(s.confidence for s in votes["long"]) / long_count
            return Signal(
                action="long",
                confidence=avg_conf,
                regime=regime,
                meta={"long_votes": long_count, "short_votes": short_count, "regime": regime},
            )
        if short_count >= self._min_votes:
            avg_conf = sum(s.confidence for s in votes["short"]) / short_count
            return Signal(
                action="short",
                confidence=avg_conf,
                regime=regime,
                meta={"long_votes": long_count, "short_votes": short_count, "regime": regime},
            )
        return Signal(
            action="hold",
            regime=regime,
            meta={"long_votes": long_count, "short_votes": short_count, "regime": regime},
        )

    def reload_weights(self) -> None:
        """Reload learned weights from disk (called by the learning cycle)."""
        self._learned_weights = self._load_learned_weights()
        logger.info("[MultiStrategy] Reloaded learned weights: %s", self._learned_weights)

    @staticmethod
    def _load_learned_weights() -> Dict[str, float]:
        """Load strategy weights from the learning system output file."""
        weights_path = Path("logs/learned_weights.json")
        if not weights_path.exists():
            return {}
        try:
            import json
            with open(weights_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            weights = data.get("weights", {})
            if weights:
                logger.info("[MultiStrategy] Loaded learned weights: %s", weights)
            return weights
        except Exception as exc:  # noqa: BLE001
            logger.warning("[MultiStrategy] Failed to load learned weights: %s", exc)
            return {}
