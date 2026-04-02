"""
MIRA_trader_C – Mean-Reversion Strategy.

Uses Bollinger Bands + RSI to detect oversold/overbought conditions.
- LONG  when price closes below the lower band AND RSI < oversold
- SHORT when price closes above the upper band AND RSI > overbought
"""
from __future__ import annotations

import pandas as pd

from indicators.technical import bollinger_bands, rsi as calc_rsi
from strategies.base import BaseStrategy, Signal


class MeanReversionStrategy(BaseStrategy):
    """Bollinger Band mean-reversion with RSI confirmation."""

    name = "mean_reversion"

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        params = self.params
        bb_period = params.get("bb_period", 20)
        bb_std = params.get("bb_std", 2.0)
        rsi_period = params.get("rsi_period", 14)
        rsi_oversold = params.get("rsi_oversold", 35)
        rsi_overbought = params.get("rsi_overbought", 65)

        close = df["close"]
        bb = bollinger_bands(close, bb_period, bb_std)
        rsi_series = calc_rsi(close, rsi_period)

        curr_close = close.iloc[-1]
        curr_rsi = rsi_series.iloc[-1]
        lower = bb["lower"].iloc[-1]
        upper = bb["upper"].iloc[-1]

        if curr_close < lower and curr_rsi < rsi_oversold:
            confidence = min((rsi_oversold - curr_rsi) / rsi_oversold, 1.0)
            return Signal(
                action="long",
                confidence=confidence,
                strategy_name=self.name,
                meta={"rsi": curr_rsi, "bb_lower": lower, "close": curr_close},
            )
        if curr_close > upper and curr_rsi > rsi_overbought:
            confidence = min((curr_rsi - rsi_overbought) / (100 - rsi_overbought), 1.0)
            return Signal(
                action="short",
                confidence=confidence,
                strategy_name=self.name,
                meta={"rsi": curr_rsi, "bb_upper": upper, "close": curr_close},
            )
        return Signal(action="hold", strategy_name=self.name)

