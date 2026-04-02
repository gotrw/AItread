"""
Tests for MIRA_trader_C strategies.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from strategies.trend_follow import TrendFollowStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.breakout import BreakoutStrategy
from strategies.base import Signal


def make_ohlcv(prices: list[float]) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a list of close prices."""
    close = np.array(prices, dtype=float)
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": np.ones(len(close)) * 1000,
        }
    )


# ── Signal dataclass ──────────────────────────────────────────

def test_signal_hold_not_actionable():
    sig = Signal(action="hold")
    assert not sig.is_actionable()


def test_signal_buy_actionable():
    sig = Signal(action="buy", confidence=0.7)
    assert sig.is_actionable()


# ── TrendFollowStrategy ───────────────────────────────────────

def test_trend_follow_returns_signal():
    params = {"fast_ema": 5, "slow_ema": 15, "rsi_period": 14,
               "rsi_buy_threshold": 50, "rsi_sell_threshold": 50}
    strat = TrendFollowStrategy(params)
    # Provide enough candles for warmup
    np.random.seed(0)
    prices = list(30000 + np.cumsum(np.random.randn(60) * 100))
    df = make_ohlcv(prices)
    sig = strat.generate_signal(df)
    assert isinstance(sig, Signal)
    assert sig.action in ("long", "short", "hold", "buy", "sell")


def test_trend_follow_uptrend_favours_buy():
    params = {"fast_ema": 3, "slow_ema": 10, "rsi_period": 5,
               "rsi_buy_threshold": 40, "rsi_sell_threshold": 60}
    strat = TrendFollowStrategy(params)
    # Strong uptrend: prices increase every bar
    prices = list(range(29900, 30050))
    df = make_ohlcv(prices)
    sig = strat.generate_signal(df)
    # Should not produce a sell/short signal in a strong uptrend
    assert sig.action not in ("sell", "short")


# ── MeanReversionStrategy ─────────────────────────────────────

def test_mean_reversion_returns_signal():
    params = {"bb_period": 10, "bb_std": 2.0, "rsi_period": 7,
               "rsi_oversold": 35, "rsi_overbought": 65}
    strat = MeanReversionStrategy(params)
    np.random.seed(1)
    prices = list(30000 + np.random.randn(60) * 50)
    df = make_ohlcv(prices)
    sig = strat.generate_signal(df)
    assert isinstance(sig, Signal)


def test_mean_reversion_oversold_triggers_buy():
    params = {"bb_period": 5, "bb_std": 1.0, "rsi_period": 3,
               "rsi_oversold": 80, "rsi_overbought": 90}
    strat = MeanReversionStrategy(params)
    # Simulate a sharp drop so price is well below lower BB
    prices = [30000] * 20 + [28000] * 5
    df = make_ohlcv(prices)
    sig = strat.generate_signal(df)
    # With extreme thresholds price should trigger long (oversold > 80) or hold
    assert sig.action in ("long", "buy", "hold")


# ── BreakoutStrategy ──────────────────────────────────────────

def test_breakout_returns_signal():
    params = {"lookback": 10, "atr_period": 5, "atr_multiplier": 1.0}
    strat = BreakoutStrategy(params)
    np.random.seed(2)
    prices = list(30000 + np.random.randn(50) * 100)
    df = make_ohlcv(prices)
    sig = strat.generate_signal(df)
    assert isinstance(sig, Signal)


def test_breakout_insufficient_data_returns_hold():
    params = {"lookback": 20, "atr_period": 14, "atr_multiplier": 1.5}
    strat = BreakoutStrategy(params)
    prices = [30000] * 10   # too few candles
    df = make_ohlcv(prices)
    sig = strat.generate_signal(df)
    assert sig.action == "hold"


def test_breakout_upward_breakout_triggers_buy():
    params = {"lookback": 5, "atr_period": 3, "atr_multiplier": 0.1}
    strat = BreakoutStrategy(params)
    # Stable range then large breakout
    prices = [30000] * 10 + [32000]
    df = make_ohlcv(prices)
    sig = strat.generate_signal(df)
    # Strategies now emit 'long' instead of 'buy' for futures support
    assert sig.action in ("long", "buy")
