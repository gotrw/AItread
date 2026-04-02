"""
MIRA_trader_C – Market Regime Detector.

Detects the current market regime for a symbol using ADX and ATR-based
volatility analysis. The regime is used by MultiStrategy to route signals
to the most appropriate sub-strategy.

Regimes
-------
- trending   : strong directional move (ADX > threshold, clear DI divergence)
- ranging    : low directional movement, price oscillating in a range
- volatile   : high volatility, ATR spike (suitable for volatility strategy)
- choppy     : low volatility but no clear trend (low ADX, low ATR)
"""
from __future__ import annotations

import logging
from typing import Literal

import pandas as pd

from indicators.technical import adx as calc_adx, atr as calc_atr, sma

logger = logging.getLogger(__name__)

RegimeType = Literal["trending", "ranging", "volatile", "choppy"]


class RegimeDetector:
    """Detect the market regime for a symbol.

    Parameters
    ----------
    params:
        Dict with keys from config ``strategy.regime``:
        - adx_period (int): period for ADX (default 14)
        - adx_trending_threshold (float): ADX > this → trending (default 25)
        - atr_period (int): period for ATR (default 14)
        - atr_volatile_multiplier (float): ATR/SMA ratio multiplier (default 1.5)
        - lookback (int): candles to consider (default 50)
    """

    def __init__(self, params: dict) -> None:
        self.adx_period: int = params.get("adx_period", 14)
        self.adx_threshold: float = params.get("adx_trending_threshold", 25.0)
        self.atr_period: int = params.get("atr_period", 14)
        self.atr_volatile_mult: float = params.get("atr_volatile_multiplier", 1.5)
        self.lookback: int = params.get("lookback", 50)

    def detect(self, df: pd.DataFrame) -> RegimeType:
        """Analyse the DataFrame and return the current market regime.

        Args:
            df: OHLCV DataFrame with at least ``self.lookback`` rows.

        Returns:
            One of 'trending', 'ranging', 'volatile', 'choppy'.
        """
        min_rows = max(self.adx_period, self.atr_period, self.lookback) + 5
        if len(df) < min_rows:
            return "choppy"

        window = df.iloc[-self.lookback :]

        adx_df = calc_adx(window, self.adx_period)
        curr_adx = float(adx_df["adx"].iloc[-1])

        atr_series = calc_atr(window, self.atr_period)
        curr_atr = float(atr_series.iloc[-1])

        # Use last N closes to compute a reference SMA for ATR normalisation
        close_sma = float(sma(window["close"], min(20, len(window))).iloc[-1])
        atr_ratio = curr_atr / close_sma if close_sma else 0.0

        # Compute average ATR ratio over the window to define "normal" volatility
        atr_window = atr_series.dropna()
        close_sma_series = sma(window["close"], min(20, len(window)))
        avg_atr_ratio = float(
            (atr_window / close_sma_series.reindex(atr_window.index).replace(0, float("nan")))
            .dropna()
            .mean()
        ) if len(atr_window) > 0 else atr_ratio

        is_volatile = atr_ratio > avg_atr_ratio * self.atr_volatile_mult
        is_trending = curr_adx > self.adx_threshold

        if is_volatile:
            regime: RegimeType = "volatile"
        elif is_trending:
            regime = "trending"
        elif curr_adx > self.adx_threshold * 0.6:
            regime = "ranging"
        else:
            regime = "choppy"

        logger.debug(
            "Regime detected: %s (ADX=%.1f, ATR_ratio=%.4f, avg_ATR_ratio=%.4f)",
            regime, curr_adx, atr_ratio, avg_atr_ratio,
        )
        return regime
