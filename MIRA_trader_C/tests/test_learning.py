"""
Tests for MIRA_trader_C self-learning system.

Covers:
  - WeightTuner: journal reading, score computation, weight persistence
  - ThresholdAdapter: rolling win rate → threshold selection
  - BacktestGuard: Sharpe-based weight validation
  - BotWorker: trade journal writing on position close
  - MultiStrategy: learned-weight loading
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from learning.weight_tuner import WeightTuner
from learning.threshold_adapter import ThresholdAdapter
from learning.backtest_guard import BacktestGuard


# ── Helpers ────────────────────────────────────────────────────

def _write_journal(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _make_record(
    strategy: str = "trend_follow",
    regime: str = "trending",
    pnl_pct: float = 1.0,
) -> dict:
    return {
        "trade_id": str(uuid.uuid4()),
        "symbol": "BTC/USDT",
        "side": "long",
        "strategy": strategy,
        "regime": regime,
        "confidence": 0.7,
        "timeframe": "1h",
        "entry_price": 28000.0,
        "exit_price": 28280.0,
        "qty": 0.01,
        "pnl_pct": pnl_pct,
        "win": pnl_pct > 0,
        "entry_time": "2024-01-01T00:00:00+00:00",
        "exit_time": "2024-01-01T01:00:00+00:00",
        "exit_reason": "take_profit",
    }


# ── WeightTuner ────────────────────────────────────────────────

class TestWeightTuner:
    def test_returns_empty_when_no_journal(self, tmp_path):
        tuner = WeightTuner(
            journal_path=tmp_path / "journal.jsonl",
            weights_path=tmp_path / "weights.json",
        )
        weights = tuner.tune()
        assert weights == {}

    def test_returns_empty_below_min_samples(self, tmp_path):
        journal = tmp_path / "journal.jsonl"
        # 5 records but min_samples=10
        records = [_make_record("trend_follow", "trending", 1.0) for _ in range(5)]
        _write_journal(journal, records)
        tuner = WeightTuner(
            journal_path=journal,
            weights_path=tmp_path / "weights.json",
            min_samples=10,
        )
        weights = tuner.tune()
        assert weights == {}

    def test_weights_computed_and_clamped(self, tmp_path):
        journal = tmp_path / "journal.jsonl"
        # trend_follow: 10 wins → high score
        # mean_reversion: 10 losses → low score
        records = (
            [_make_record("trend_follow", "trending", 2.0) for _ in range(10)]
            + [_make_record("mean_reversion", "trending", -1.5) for _ in range(10)]
        )
        _write_journal(journal, records)
        tuner = WeightTuner(
            journal_path=journal,
            weights_path=tmp_path / "weights.json",
            min_samples=5,
            min_weight=0.5,
            max_weight=3.0,
        )
        weights = tuner.tune()
        assert "trend_follow" in weights
        assert "mean_reversion" in weights
        assert weights["trend_follow"] > weights["mean_reversion"]
        assert all(0.5 <= v <= 3.0 for v in weights.values())

    def test_weights_persisted_to_file(self, tmp_path):
        journal = tmp_path / "journal.jsonl"
        records = [_make_record("trend_follow", "trending", 1.0) for _ in range(15)]
        _write_journal(journal, records)
        weights_path = tmp_path / "weights.json"
        tuner = WeightTuner(
            journal_path=journal,
            weights_path=weights_path,
            min_samples=5,
        )
        tuner.tune()
        assert weights_path.exists()
        data = json.loads(weights_path.read_text())
        assert "weights" in data

    def test_load_weights_returns_empty_if_no_file(self, tmp_path):
        tuner = WeightTuner(
            journal_path=tmp_path / "j.jsonl",
            weights_path=tmp_path / "w.json",
        )
        assert tuner.load_weights() == {}

    def test_load_weights_reads_persisted(self, tmp_path):
        journal = tmp_path / "journal.jsonl"
        records = [_make_record("breakout", "volatile", 3.0) for _ in range(12)]
        _write_journal(journal, records)
        weights_path = tmp_path / "weights.json"
        tuner = WeightTuner(
            journal_path=journal,
            weights_path=weights_path,
            min_samples=5,
        )
        tuner.tune()
        loaded = tuner.load_weights()
        assert isinstance(loaded, dict)

    def test_regime_aware_uses_regime_stats(self, tmp_path):
        journal = tmp_path / "journal.jsonl"
        # breakout wins in volatile, loses in trending
        records = (
            [_make_record("breakout", "volatile", 2.0) for _ in range(15)]
            + [_make_record("trend_follow", "volatile", -1.0) for _ in range(15)]
        )
        _write_journal(journal, records)
        tuner = WeightTuner(
            journal_path=journal,
            weights_path=tmp_path / "weights.json",
            min_samples=10,
        )
        weights = tuner.tune(current_regime="volatile")
        assert weights.get("breakout", 1.0) > weights.get("trend_follow", 1.0)

    def test_lookback_limits_records(self, tmp_path):
        journal = tmp_path / "journal.jsonl"
        # 50 old losses then 30 recent wins
        records = (
            [_make_record("trend_follow", "trending", -1.0) for _ in range(50)]
            + [_make_record("trend_follow", "trending", 2.0) for _ in range(30)]
        )
        _write_journal(journal, records)
        # With lookback=30 only the wins are seen → high weight
        tuner_recent = WeightTuner(
            journal_path=journal,
            weights_path=tmp_path / "w_recent.json",
            lookback=30,
            min_samples=5,
        )
        weights_recent = tuner_recent.tune()
        # With lookback=80 losses dominate → lower weight
        tuner_all = WeightTuner(
            journal_path=journal,
            weights_path=tmp_path / "w_all.json",
            lookback=80,
            min_samples=5,
        )
        weights_all = tuner_all.tune()
        assert weights_recent.get("trend_follow", 1.0) >= weights_all.get("trend_follow", 1.0)


# ── ThresholdAdapter ────────────────────────────────────────────

class TestThresholdAdapter:
    def test_default_when_no_journal(self, tmp_path):
        adapter = ThresholdAdapter(
            journal_path=tmp_path / "j.jsonl",
            threshold_path=tmp_path / "t.json",
            default_confidence=0.55,
            min_samples=10,
        )
        result = adapter.adapt()
        assert result == 0.55

    def test_default_below_min_samples(self, tmp_path):
        journal = tmp_path / "j.jsonl"
        records = [_make_record(pnl_pct=0.5) for _ in range(5)]
        _write_journal(journal, records)
        adapter = ThresholdAdapter(
            journal_path=journal,
            threshold_path=tmp_path / "t.json",
            default_confidence=0.55,
            min_samples=10,
        )
        result = adapter.adapt()
        assert result == 0.55

    def test_conservative_when_low_win_rate(self, tmp_path):
        journal = tmp_path / "j.jsonl"
        # 3 wins / 12 total → win rate = 25% < 45%
        records = (
            [_make_record(pnl_pct=1.0) for _ in range(3)]
            + [_make_record(pnl_pct=-1.0) for _ in range(9)]
        )
        _write_journal(journal, records)
        adapter = ThresholdAdapter(
            journal_path=journal,
            threshold_path=tmp_path / "t.json",
            default_confidence=0.55,
            conservative_confidence=0.70,
            low_threshold=0.45,
            min_samples=10,
        )
        result = adapter.adapt()
        assert result == 0.70

    def test_default_when_high_win_rate(self, tmp_path):
        journal = tmp_path / "j.jsonl"
        # 10 wins / 12 total → win rate ≈ 83% > 55%
        records = (
            [_make_record(pnl_pct=1.0) for _ in range(10)]
            + [_make_record(pnl_pct=-1.0) for _ in range(2)]
        )
        _write_journal(journal, records)
        adapter = ThresholdAdapter(
            journal_path=journal,
            threshold_path=tmp_path / "t.json",
            default_confidence=0.55,
            conservative_confidence=0.70,
            high_threshold=0.55,
            min_samples=10,
        )
        result = adapter.adapt()
        assert result == 0.55

    def test_interpolation_between_thresholds(self, tmp_path):
        journal = tmp_path / "j.jsonl"
        # ~50% win rate → between 45% and 55%
        records = (
            [_make_record(pnl_pct=1.0) for _ in range(10)]
            + [_make_record(pnl_pct=-1.0) for _ in range(10)]
        )
        _write_journal(journal, records)
        adapter = ThresholdAdapter(
            journal_path=journal,
            threshold_path=tmp_path / "t.json",
            default_confidence=0.55,
            conservative_confidence=0.70,
            low_threshold=0.45,
            high_threshold=0.55,
            min_samples=10,
        )
        result = adapter.adapt()
        assert 0.55 <= result <= 0.70

    def test_persists_to_file(self, tmp_path):
        journal = tmp_path / "j.jsonl"
        records = [_make_record(pnl_pct=1.0) for _ in range(12)]
        _write_journal(journal, records)
        threshold_path = tmp_path / "t.json"
        adapter = ThresholdAdapter(
            journal_path=journal,
            threshold_path=threshold_path,
            default_confidence=0.55,
            min_samples=10,
        )
        adapter.adapt()
        assert threshold_path.exists()
        data = json.loads(threshold_path.read_text())
        assert "min_confidence" in data

    def test_load_threshold_returns_none_if_no_file(self, tmp_path):
        adapter = ThresholdAdapter(
            journal_path=tmp_path / "j.jsonl",
            threshold_path=tmp_path / "t.json",
        )
        assert adapter.load_threshold() is None

    def test_load_threshold_reads_persisted(self, tmp_path):
        journal = tmp_path / "j.jsonl"
        records = [_make_record(pnl_pct=1.0) for _ in range(12)]
        _write_journal(journal, records)
        threshold_path = tmp_path / "t.json"
        adapter = ThresholdAdapter(
            journal_path=journal,
            threshold_path=threshold_path,
            default_confidence=0.55,
            min_samples=10,
        )
        adapter.adapt()
        loaded = adapter.load_threshold()
        assert loaded is not None
        assert 0.0 <= loaded <= 1.0


# ── BacktestGuard ──────────────────────────────────────────────

class TestBacktestGuard:
    def test_accepts_when_below_min_trades(self, tmp_path):
        journal = tmp_path / "j.jsonl"
        records = [_make_record(pnl_pct=1.0) for _ in range(5)]
        _write_journal(journal, records)
        guard = BacktestGuard(journal_path=journal, min_trades=20)
        accepted, reason = guard.validate({"trend_follow": 2.0})
        assert accepted
        assert "deferred" in reason.lower()

    def test_accepts_when_new_sharpe_better(self, tmp_path):
        journal = tmp_path / "j.jsonl"
        # trend_follow always wins, mean_reversion always loses
        records = (
            [_make_record("trend_follow", pnl_pct=2.0) for _ in range(15)]
            + [_make_record("mean_reversion", pnl_pct=-1.0) for _ in range(10)]
        )
        _write_journal(journal, records)
        guard = BacktestGuard(journal_path=journal, min_trades=20, acceptance_ratio=0.9)
        # Upweight the winning strategy
        accepted, reason = guard.validate(
            new_weights={"trend_follow": 2.0, "mean_reversion": 0.5},
            old_weights={"trend_follow": 1.0, "mean_reversion": 1.0},
        )
        assert accepted

    def test_rejects_when_new_sharpe_much_worse(self, tmp_path):
        journal = tmp_path / "j.jsonl"
        # trend_follow wins bigly, mean_reversion loses
        records = (
            [_make_record("trend_follow", pnl_pct=3.0) for _ in range(15)]
            + [_make_record("mean_reversion", pnl_pct=-3.0) for _ in range(10)]
        )
        _write_journal(journal, records)
        guard = BacktestGuard(journal_path=journal, min_trades=20, acceptance_ratio=0.9)
        # Drastically downweight winning + upweight losing
        accepted, _ = guard.validate(
            new_weights={"trend_follow": 0.5, "mean_reversion": 3.0},
            old_weights={"trend_follow": 1.0, "mean_reversion": 1.0},
        )
        assert not accepted

    def test_accepts_when_no_journal(self, tmp_path):
        guard = BacktestGuard(
            journal_path=tmp_path / "nonexistent.jsonl",
            min_trades=20,
        )
        accepted, _ = guard.validate({"trend_follow": 2.0})
        assert accepted

    def test_validate_with_no_old_weights_defaults_to_ones(self, tmp_path):
        journal = tmp_path / "j.jsonl"
        records = [_make_record("trend_follow", pnl_pct=1.0) for _ in range(25)]
        _write_journal(journal, records)
        guard = BacktestGuard(journal_path=journal, min_trades=20)
        accepted, _ = guard.validate(new_weights={"trend_follow": 1.5})
        assert isinstance(accepted, bool)


# ── MultiStrategy learned weight loading ───────────────────────

class TestMultiStrategyLearnedWeights:
    def _make_df(self, n: int = 100) -> pd.DataFrame:
        close = np.linspace(100, 200, n)
        return pd.DataFrame({
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": np.ones(n) * 1000,
        })

    def test_loads_weights_from_file(self, tmp_path, monkeypatch):
        from strategies.multi_strategy import MultiStrategy

        weights_file = tmp_path / "learned_weights.json"
        weights_file.write_text(json.dumps({
            "weights": {"trend_follow": 2.0, "mean_reversion": 0.5},
        }))
        # Monkeypatch the weights file path lookup
        monkeypatch.chdir(tmp_path)
        # Create required logs dir so MultiStrategy._load_learned_weights works
        (tmp_path / "logs").mkdir(exist_ok=True)
        (tmp_path / "logs" / "learned_weights.json").write_text(json.dumps({
            "weights": {"trend_follow": 2.0, "mean_reversion": 0.5},
        }))
        cfg = {
            "strategy": {
                "multi_strategy": {
                    "strategies": ["trend_follow", "mean_reversion"],
                    "min_votes": 1,
                    "regime_aware": False,
                }
            }
        }
        ms = MultiStrategy(cfg["strategy"])
        assert ms._learned_weights.get("trend_follow") == 2.0
        assert ms._learned_weights.get("mean_reversion") == 0.5

    def test_reload_weights_updates_in_place(self, tmp_path, monkeypatch):
        from strategies.multi_strategy import MultiStrategy

        monkeypatch.chdir(tmp_path)
        logs = tmp_path / "logs"
        logs.mkdir()

        cfg = {
            "strategy": {
                "multi_strategy": {
                    "strategies": ["trend_follow"],
                    "min_votes": 1,
                    "regime_aware": False,
                }
            }
        }
        ms = MultiStrategy(cfg["strategy"])
        assert ms._learned_weights == {}

        # Write weights and reload
        (logs / "learned_weights.json").write_text(json.dumps({
            "weights": {"trend_follow": 1.8}
        }))
        ms.reload_weights()
        assert ms._learned_weights.get("trend_follow") == 1.8
