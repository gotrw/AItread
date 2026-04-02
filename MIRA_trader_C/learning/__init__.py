"""
MIRA_trader_C – Learning System.

Provides a feedback loop from completed trades back into strategy weights
and signal confidence thresholds, allowing the bot to improve its win rate
over time without requiring complex machine-learning infrastructure.

Components
----------
weight_tuner.py
    Reads trade_journal.jsonl and computes per-strategy / per-regime win rates.
    Adjusts strategy vote-weights in MultiStrategy so that historically
    profitable strategies receive more influence.

threshold_adapter.py
    Monitors rolling win rate and raises the minimum confidence threshold
    when performance is poor (conservative mode) or lowers it when good.

backtest_guard.py
    Validates that proposed weight changes would not have made recent
    performance worse before committing them.
"""

from .weight_tuner import WeightTuner
from .threshold_adapter import ThresholdAdapter
from .backtest_guard import BacktestGuard

__all__ = ["WeightTuner", "ThresholdAdapter", "BacktestGuard"]
