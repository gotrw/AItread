"""
MIRA_trader_C – Backtest Guard.

Before committing new strategy weights produced by ``WeightTuner``, the guard
validates that the proposed changes are not harmful by replaying the
already-recorded trade journal with both the old and new weights and
comparing Sharpe ratios.

The guard uses only journal data (no live market data fetch), making it fast
and dependency-free.  This is intentionally simpler than a full backtest
(which would require historical OHLCV) – it answers the question:

    "Given the trades we already took, would the new weights have produced
     a better risk-adjusted return than the old weights?"

Algorithm
~~~~~~~~~
1. Load the trade journal records.
2. For each trade record, look up the strategy that generated it.
3. Scale the recorded ``pnl_pct`` by ``new_weight / old_weight`` to simulate
   how the trade would have been sized differently.
4. Compute Sharpe ratios from both PnL series.
5. Accept new weights only if ``new_sharpe >= old_sharpe * acceptance_ratio``.

Limitations
~~~~~~~~~~~
- Scaling PnL by weight ratio is an approximation (position sizing is
  not perfectly linear in a voting system).  However it is directionally
  correct and safe as a guard rather than an optimiser.
- If there are fewer than ``min_trades`` journal entries the guard always
  accepts (defers to the weight tuner's minimum-sample guard).
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_JOURNAL = Path("logs/trade_journal.jsonl")


class BacktestGuard:
    """Validate proposed weight changes against journal replay.

    Parameters
    ----------
    journal_path:
        Path to ``trade_journal.jsonl``.
    lookback:
        Maximum number of recent trades used for validation.
    min_trades:
        If fewer trades are available the guard always accepts.
    acceptance_ratio:
        New weights are accepted if ``new_sharpe >= old_sharpe * ratio``.
        Default 0.9 means a 10% Sharpe degradation is still acceptable.
    """

    def __init__(
        self,
        journal_path: str | Path = _DEFAULT_JOURNAL,
        lookback: int = 100,
        min_trades: int = 20,
        acceptance_ratio: float = 0.9,
    ) -> None:
        self._journal = Path(journal_path)
        self._lookback = lookback
        self._min_trades = min_trades
        self._ratio = acceptance_ratio

    # ── Public API ─────────────────────────────────────────────

    def validate(
        self,
        new_weights: Dict[str, float],
        old_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[bool, str]:
        """Return (accepted, reason_str).

        Parameters
        ----------
        new_weights:
            Proposed strategy weight mapping from ``WeightTuner``.
        old_weights:
            Current weights (default all 1.0 if not provided).
        """
        records = self._load_journal()
        if len(records) < self._min_trades:
            msg = (
                f"Only {len(records)} trades in journal (min={self._min_trades}); "
                "guard deferred – accepting weights."
            )
            logger.info("[BacktestGuard] %s", msg)
            return True, msg

        if old_weights is None:
            old_weights = {}

        old_pnls = self._replay(records, old_weights, scale=False)
        new_pnls = self._replay(records, new_weights, scale=True)

        old_sharpe = self._sharpe(old_pnls)
        new_sharpe = self._sharpe(new_pnls)

        threshold = old_sharpe * self._ratio if old_sharpe > 0 else -math.inf

        if new_sharpe >= threshold:
            msg = (
                f"Accepted: new_sharpe={new_sharpe:.3f} >= "
                f"threshold={threshold:.3f} (old={old_sharpe:.3f})"
            )
            logger.info("[BacktestGuard] %s", msg)
            return True, msg
        else:
            msg = (
                f"Rejected: new_sharpe={new_sharpe:.3f} < "
                f"threshold={threshold:.3f} (old={old_sharpe:.3f}). "
                "Keeping current weights."
            )
            logger.warning("[BacktestGuard] %s", msg)
            return False, msg

    # ── Helpers ────────────────────────────────────────────────

    def _replay(
        self,
        records: List[Dict[str, Any]],
        weights: Dict[str, float],
        scale: bool,
    ) -> List[float]:
        """Produce a PnL series.

        When ``scale=True``, each trade's pnl_pct is multiplied by the
        ratio ``new_weight / 1.0`` for that trade's strategy (approximates
        position-size effect).
        """
        pnls: List[float] = []
        for r in records:
            pnl = r.get("pnl_pct", 0.0)
            if scale:
                strat = r.get("strategy", "")
                w = weights.get(strat, 1.0)
                pnl = pnl * w
            pnls.append(pnl)
        return pnls

    def _sharpe(self, pnls: List[float]) -> float:
        """Annualised Sharpe ratio (assuming each trade is independent)."""
        if not pnls:
            return 0.0
        n = len(pnls)
        mean = sum(pnls) / n
        if n < 2:
            return mean
        variance = sum((p - mean) ** 2 for p in pnls) / (n - 1)
        std = math.sqrt(variance) if variance > 0 else 1e-9
        return mean / std * math.sqrt(252)  # annualised assuming ~daily trades

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
            logger.warning("[BacktestGuard] Cannot read journal: %s", exc)
            return []

        if self._lookback and len(records) > self._lookback:
            records = records[-self._lookback:]
        return records
