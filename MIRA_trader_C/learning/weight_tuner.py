"""
MIRA_trader_C – Strategy Weight Tuner.

Reads the trade journal and computes per-strategy / per-regime performance
metrics (win rate, average PnL).  Produces a new weight mapping that gives
more influence to historically profitable strategies in the current regime.

Weights are persisted to ``logs/learned_weights.json`` so that
``MultiStrategy`` can load them on the next initialisation (or
live-reload via the periodic learning cycle).

Weight bounds
~~~~~~~~~~~~~
Each strategy weight is clamped to [``min_weight``, ``max_weight``]
(defaults 0.5–3.0) to prevent a single bad run from completely
silencing a strategy.

Minimum sample guard
~~~~~~~~~~~~~~~~~~~~
A strategy/regime combination must have at least ``min_samples`` trades
before its weight is adjusted.  Until then it keeps its default weight (1.0).
"""
from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_JOURNAL = Path("logs/trade_journal.jsonl")
_DEFAULT_WEIGHTS_FILE = Path("logs/learned_weights.json")

# Regime → list of strategies that are generally a good match.
# Used to give a small prior boost when historical data is sparse.
_REGIME_AFFINITY: Dict[str, List[str]] = {
    "trending":  ["trend_follow", "breakout"],
    "ranging":   ["mean_reversion"],
    "volatile":  ["volatility", "breakout"],
    "choppy":    ["volatility", "mean_reversion"],
}


class WeightTuner:
    """Compute and persist updated strategy weights from trade history.

    Parameters
    ----------
    journal_path:
        Path to ``trade_journal.jsonl``.
    weights_path:
        Where to write ``learned_weights.json``.
    lookback:
        Maximum number of recent trades to analyse (0 = all).
    min_samples:
        Minimum trades per (strategy, regime) pair before adjusting weight.
    min_weight, max_weight:
        Clamp bounds for individual strategy weights.
    win_rate_weight, pnl_weight:
        Relative importance of win-rate vs average PnL in the score.
    """

    def __init__(
        self,
        journal_path: str | Path = _DEFAULT_JOURNAL,
        weights_path: str | Path = _DEFAULT_WEIGHTS_FILE,
        lookback: int = 100,
        min_samples: int = 10,
        min_weight: float = 0.5,
        max_weight: float = 3.0,
        win_rate_weight: float = 0.6,
        pnl_weight: float = 0.4,
    ) -> None:
        self._journal = Path(journal_path)
        self._weights_path = Path(weights_path)
        self._lookback = lookback
        self._min_samples = min_samples
        self._min_weight = min_weight
        self._max_weight = max_weight
        self._wr_w = win_rate_weight
        self._pnl_w = pnl_weight

    # ── Public API ─────────────────────────────────────────────

    def tune(self, current_regime: str = "") -> Dict[str, float]:
        """Compute and save updated weights.  Returns the weight mapping.

        Parameters
        ----------
        current_regime:
            The regime currently detected by the live bot.  When provided,
            the tuner uses regime-specific statistics if available; falls
            back to global (regime-agnostic) stats otherwise.

        Returns
        -------
        dict mapping strategy_name → float vote-weight.
        """
        records = self._load_journal()
        if not records:
            logger.info("[WeightTuner] No journal records found; keeping default weights.")
            return {}

        # Collect per-(strategy, regime) and global-per-strategy stats
        regime_stats: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )  # regime → strategy → [pnl_pct, ...]
        global_stats: Dict[str, List[float]] = defaultdict(list)  # strategy → [pnl_pct, ...]

        for r in records:
            strat = r.get("strategy", "")
            regime = r.get("regime", "")
            pnl = r.get("pnl_pct", 0.0)
            if not strat:
                continue
            global_stats[strat].append(pnl)
            if regime:
                regime_stats[regime][strat].append(pnl)

        # Compute weights using regime-specific stats when available
        stats_to_use: Dict[str, List[float]] = {}
        if current_regime and current_regime in regime_stats:
            for strat, pnls in regime_stats[current_regime].items():
                if len(pnls) >= self._min_samples:
                    stats_to_use[strat] = pnls
                    logger.debug(
                        "[WeightTuner] Using regime=%s stats for %s (%d trades)",
                        current_regime, strat, len(pnls),
                    )

        # Fall back to global stats for strategies without regime data
        for strat, pnls in global_stats.items():
            if strat not in stats_to_use and len(pnls) >= self._min_samples:
                stats_to_use[strat] = pnls

        if not stats_to_use:
            logger.info(
                "[WeightTuner] Not enough samples (min=%d) to tune weights.",
                self._min_samples,
            )
            return {}

        scores = {s: self._score(pnls) for s, pnls in stats_to_use.items()}
        weights = self._scores_to_weights(scores)

        # Persist
        os.makedirs(self._weights_path.parent, exist_ok=True)
        payload: Dict[str, Any] = {
            "weights": weights,
            "regime": current_regime,
            "samples": {s: len(p) for s, p in stats_to_use.items()},
            "scores": {s: round(v, 4) for s, v in scores.items()},
        }
        with open(self._weights_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

        logger.info("[WeightTuner] Saved weights %s → %s", weights, self._weights_path)
        return weights

    def load_weights(self) -> Dict[str, float]:
        """Load previously persisted weights (empty dict if none exist)."""
        if not self._weights_path.exists():
            return {}
        try:
            with open(self._weights_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data.get("weights", {})
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("[WeightTuner] Failed to load weights: %s", exc)
            return {}

    # ── Helpers ────────────────────────────────────────────────

    def _load_journal(self) -> List[Dict[str, Any]]:
        if not self._journal.exists():
            return []
        records: List[Dict[str, Any]] = []
        try:
            with open(self._journal, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except OSError as exc:
            logger.warning("[WeightTuner] Cannot read journal: %s", exc)
            return []

        # Keep only the most recent N trades
        if self._lookback and len(records) > self._lookback:
            records = records[-self._lookback:]
        return records

    def _score(self, pnls: List[float]) -> float:
        """Compute a composite score in [0, 1] from a list of PnL percentages."""
        if not pnls:
            return 0.5
        win_rate = sum(1 for p in pnls if p > 0) / len(pnls)
        avg_pnl = sum(pnls) / len(pnls)
        # Normalise avg_pnl: clamp to [-5%, +5%] then scale to [0, 1]
        pnl_score = max(0.0, min(1.0, (avg_pnl + 5.0) / 10.0))
        return self._wr_w * win_rate + self._pnl_w * pnl_score

    def _scores_to_weights(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Convert raw scores to clamped vote-weights."""
        if not scores:
            return {}
        avg_score = sum(scores.values()) / len(scores)
        if avg_score == 0:
            return {s: 1.0 for s in scores}
        weights: Dict[str, float] = {}
        for strat, score in scores.items():
            raw = score / avg_score  # relative weight around 1.0
            clamped = max(self._min_weight, min(self._max_weight, raw))
            weights[strat] = round(clamped, 3)
        return weights
