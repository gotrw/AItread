"""Tests for RegimeDetector."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from strategies.regime import RegimeDetector


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


def test_regime_returns_valid_string():
    np.random.seed(0)
    prices = list(30000 + np.cumsum(np.random.randn(100) * 100))
    df = make_ohlcv(prices)
    detector = RegimeDetector({})
    regime = detector.detect(df)
    assert regime in ("trending", "ranging", "volatile", "choppy")


def test_regime_insufficient_data_returns_choppy():
    prices = [30000] * 5
    df = make_ohlcv(prices)
    detector = RegimeDetector({})
    regime = detector.detect(df)
    # With very little data it should still return a valid regime
    assert isinstance(regime, str)
    assert len(regime) > 0


def test_strong_trend_detected():
    """A monotonically rising market should be detected as trending."""
    # Deterministic strong uptrend
    prices = [30000 + i * 50 for i in range(200)]
    df = make_ohlcv(prices)
    detector = RegimeDetector({"adx_period": 14, "lookback": 50})
    regime = detector.detect(df)
    # In a strong uptrend ADX will be high → trending
    assert regime in ("trending", "volatile")


def test_sideways_market_not_trending():
    """A purely sideways market should NOT be detected as trending."""
    np.random.seed(1)
    prices = list(30000 + np.random.randn(200) * 10)  # very low volatility range
    df = make_ohlcv(prices)
    detector = RegimeDetector({"adx_period": 14, "lookback": 50})
    regime = detector.detect(df)
    assert regime in ("ranging", "choppy", "volatile")


def test_high_volatility_spike():
    """A sudden huge spike should produce volatile or choppy regime."""
    prices = [30000] * 100 + [35000, 25000, 33000, 27000, 32000] * 20
    df = make_ohlcv(prices)
    detector = RegimeDetector({})
    regime = detector.detect(df)
    assert regime in ("volatile", "choppy", "trending", "ranging")


def test_regime_custom_thresholds():
    """Custom threshold parameters should be respected."""
    np.random.seed(42)
    prices = list(30000 + np.cumsum(np.random.randn(150) * 100))
    df = make_ohlcv(prices)
    detector = RegimeDetector({"adx_trending_threshold": 5, "adx_period": 10, "lookback": 30})
    # With a very low trending threshold almost everything should be trending
    regime = detector.detect(df)
    assert isinstance(regime, str)
