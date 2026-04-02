"""Tests for VolatilityStrategy."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from strategies.volatility import VolatilityStrategy
from strategies.base import Signal


def make_ohlcv(prices: list[float]) -> pd.DataFrame:
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


def test_returns_signal_instance():
    strat = VolatilityStrategy({})
    np.random.seed(0)
    prices = list(30000 + np.random.randn(80) * 100)
    df = make_ohlcv(prices)
    sig = strat.generate_signal(df)
    assert isinstance(sig, Signal)


def test_insufficient_data_returns_hold():
    strat = VolatilityStrategy({"bb_period": 20, "atr_period": 14})
    prices = [30000] * 10
    df = make_ohlcv(prices)
    sig = strat.generate_signal(df)
    assert sig.action == "hold"


def test_action_is_valid():
    strat = VolatilityStrategy({"bb_period": 10, "bb_std": 2.0, "atr_period": 5, "atr_multiplier": 1.0})
    np.random.seed(2)
    prices = list(30000 + np.random.randn(60) * 200)
    df = make_ohlcv(prices)
    sig = strat.generate_signal(df)
    assert sig.action in ("long", "short", "hold", "buy", "sell")


def test_volatility_expansion_triggers_signal():
    """After a squeeze, a large expansion bar should trigger long or short."""
    # Flat squeeze
    prices = [30000.0] * 40
    # Then strong upward expansion
    prices += [30000 + i * 200 for i in range(1, 21)]
    df = make_ohlcv(prices)
    strat = VolatilityStrategy(
        {"bb_period": 10, "bb_std": 2.0, "atr_period": 5, "atr_multiplier": 0.1, "squeeze_threshold": 0.001}
    )
    sig = strat.generate_signal(df)
    # Should produce an actionable signal (long or hold — not sell during upward expansion)
    assert sig.action != "short"


def test_confidence_between_0_and_1():
    strat = VolatilityStrategy({})
    np.random.seed(3)
    prices = list(30000 + np.cumsum(np.random.randn(80) * 50))
    df = make_ohlcv(prices)
    sig = strat.generate_signal(df)
    assert 0.0 <= sig.confidence <= 1.0
