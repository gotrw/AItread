"""
Tests for MIRA_trader_C technical indicators.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from indicators.technical import ema, sma, rsi, macd, bollinger_bands, atr


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def price_series() -> pd.Series:
    """Synthetic close-price series (100 bars)."""
    np.random.seed(42)
    prices = 30000 + np.cumsum(np.random.randn(100) * 100)
    return pd.Series(prices, name="close")


@pytest.fixture
def ohlcv_df(price_series: pd.Series) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame."""
    close = price_series.values
    df = pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": np.random.uniform(100, 500, len(close)),
        }
    )
    return df


# ── EMA ───────────────────────────────────────────────────────

def test_ema_length(price_series):
    result = ema(price_series, 9)
    assert len(result) == len(price_series)


def test_ema_first_value_not_nan(price_series):
    result = ema(price_series, 9)
    # EMA with adjust=False returns a valid value for all rows
    assert not result.isna().any()


def test_ema_trend(price_series):
    fast = ema(price_series, 5)
    slow = ema(price_series, 20)
    # Fast EMA should be more responsive (not identical to slow)
    assert not (fast == slow).all()


# ── SMA ───────────────────────────────────────────────────────

def test_sma_length(price_series):
    result = sma(price_series, 10)
    assert len(result) == len(price_series)


def test_sma_first_n_nan(price_series):
    period = 10
    result = sma(price_series, period)
    assert result.iloc[: period - 1].isna().all()
    assert not result.iloc[period - 1 :].isna().any()


# ── RSI ───────────────────────────────────────────────────────

def test_rsi_range(price_series):
    result = rsi(price_series, 14)
    valid = result.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_rsi_length(price_series):
    result = rsi(price_series, 14)
    assert len(result) == len(price_series)


# ── MACD ──────────────────────────────────────────────────────

def test_macd_columns(price_series):
    result = macd(price_series)
    assert set(result.columns) == {"macd", "signal", "histogram"}


def test_macd_histogram_equals_diff(price_series):
    result = macd(price_series)
    diff = (result["macd"] - result["signal"]).round(8)
    hist = result["histogram"].round(8)
    pd.testing.assert_series_equal(diff, hist, check_names=False)


# ── Bollinger Bands ───────────────────────────────────────────

def test_bollinger_columns(price_series):
    result = bollinger_bands(price_series)
    assert {"upper", "middle", "lower", "width", "pct_b"}.issubset(result.columns)


def test_bollinger_upper_above_lower(price_series):
    result = bollinger_bands(price_series)
    valid = result.dropna()
    assert (valid["upper"] > valid["lower"]).all()


def test_bollinger_middle_between_bands(price_series):
    result = bollinger_bands(price_series)
    valid = result.dropna()
    assert (valid["middle"] <= valid["upper"]).all()
    assert (valid["middle"] >= valid["lower"]).all()


# ── ATR ───────────────────────────────────────────────────────

def test_atr_positive(ohlcv_df):
    result = atr(ohlcv_df, 14)
    valid = result.dropna()
    assert (valid > 0).all()


def test_atr_length(ohlcv_df):
    result = atr(ohlcv_df, 14)
    assert len(result) == len(ohlcv_df)
