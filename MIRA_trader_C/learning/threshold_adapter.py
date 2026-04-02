"""
MIRA_trader_C – Confidence Threshold Adapter.

Monitors the rolling win rate from ``trade_journal.jsonl`` and automatically
adjusts the ``min_confidence`` threshold used by the signal scanner and
strategy entry gate:

* When ``rolling_win_rate < low_threshold`` → raise min_confidence to
  ``conservative_confidence`` (filter out marginal signals).
* When ``rolling_win_rate > high_threshold`` → restore min_confidence to
  the default value from config.
* Otherwise keep the current value.

The computed threshold is persisted to ``logs/learned_threshold.json`` and
read by the signal scanner / bot worker on each evaluation.

This prevents the bot from over-trading during losing streaks.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_JOURNAL = Path("logs/trade_journal.jsonl")
_DEFAULT_THRESHOLD_FILE = Path("logs/learned_threshold.json")


class ThresholdAdapter:
    """Adapt signal confidence threshold based on recent win rate.

    Parameters
    ----------
    journal_path:
        Path to ``trade_journal.jsonl``.
    threshold_path:
        Where to write the learned threshold JSON.
    lookback:
        Number of recent trades to use for rolling win rate.
    default_confidence:
        Baseline ``min_confidence`` from config (e.g. 0.55).
    conservative_confidence:
        Raised threshold used during losing streaks (e.g. 0.70).
    low_threshold:
        Win rate below this triggers conservative mode (default 0.45).
    high_threshold:
        Win rate above this restores default mode (default 0.55).
    min_samples:
        Minimum trades before adapting the threshold.
    """

    def __init__(
        self,
        journal_path: str | Path = _DEFAULT_JOURNAL,
        threshold_path: str | Path = _DEFAULT_THRESHOLD_FILE,
        lookback: int = 30,
        default_confidence: float = 0.55,
        conservative_confidence: float = 0.70,
        low_threshold: float = 0.45,
        high_threshold: float = 0.55,
        min_samples: int = 10,
    ) -> None:
        self._journal = Path(journal_path)
        self._threshold_path = Path(threshold_path)
        self._lookback = lookback
        self._default_conf = default_confidence
        self._conservative_conf = conservative_confidence
        self._low_wr = low_threshold
        self._high_wr = high_threshold
        self._min_samples = min_samples

    # ── Public API ─────────────────────────────────────────────

    def adapt(self) -> float:
        """Compute and persist the recommended min_confidence.

        Returns
        -------
        float – the recommended ``min_confidence`` value.
        """
        win_rate, n_trades = self._rolling_win_rate()

        if n_trades < self._min_samples:
            logger.info(
                "[ThresholdAdapter] Only %d trades (min=%d); keeping default %.2f",
                n_trades, self._min_samples, self._default_conf,
            )
            recommended = self._default_conf
            mode = "default"
        elif win_rate < self._low_wr:
            recommended = self._conservative_conf
            mode = "conservative"
            logger.info(
                "[ThresholdAdapter] Win rate %.1f%% < %.0f%% → conservative mode "
                "(min_confidence=%.2f)",
                win_rate * 100, self._low_wr * 100, recommended,
            )
        elif win_rate > self._high_wr:
            recommended = self._default_conf
            mode = "default"
            logger.info(
                "[ThresholdAdapter] Win rate %.1f%% > %.0f%% → normal mode "
                "(min_confidence=%.2f)",
                win_rate * 100, self._high_wr * 100, recommended,
            )
        else:
            # In-between: interpolate linearly
            span = max(self._high_wr - self._low_wr, 1e-9)
            frac = (win_rate - self._low_wr) / span  # 0 → conservative, 1 → default
            recommended = round(
                self._conservative_conf + frac * (self._default_conf - self._conservative_conf),
                4,
            )
            mode = "interpolated"
            logger.info(
                "[ThresholdAdapter] Win rate %.1f%% → interpolated min_confidence=%.3f",
                win_rate * 100, recommended,
            )

        # Persist
        os.makedirs(self._threshold_path.parent, exist_ok=True)
        payload: Dict[str, Any] = {
            "min_confidence": recommended,
            "mode": mode,
            "rolling_win_rate": round(win_rate, 4),
            "n_trades": n_trades,
        }
        with open(self._threshold_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

        return recommended

    def load_threshold(self) -> Optional[float]:
        """Load previously persisted threshold (None if no file exists)."""
        if not self._threshold_path.exists():
            return None
        try:
            with open(self._threshold_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return float(data.get("min_confidence", self._default_conf))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("[ThresholdAdapter] Failed to load threshold: %s", exc)
            return None

    # ── Helpers ────────────────────────────────────────────────

    def _rolling_win_rate(self) -> tuple[float, int]:
        """Return (win_rate, n_trades) over the last ``lookback`` trades."""
        records = self._load_journal()
        if not records:
            return 0.5, 0
        wins = sum(1 for r in records if r.get("pnl_pct", 0.0) > 0)
        return wins / len(records), len(records)

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
            logger.warning("[ThresholdAdapter] Cannot read journal: %s", exc)
            return []

        if self._lookback and len(records) > self._lookback:
            records = records[-self._lookback:]
        return records
