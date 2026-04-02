"""
MIRA_trader_C – Breakout Strategy.

Detects breakouts above recent N-candle range high or below range low,
with an ATR-based buffer to avoid false breaks.
- BUY  when close > range_high + ATR * multiplier
- SELL when close < range_low  - ATR * multiplier
"""
from __future__ import annotations

import pandas as pd

from indicators.technical import atr as calc_atr
from strategies.base import BaseStrategy, Signal


class BreakoutStrategy(BaseStrategy):
    """Range breakout with ATR confirmation."""

    name = "breakout"

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        params = self.params
        lookback = params.get("lookback", 20)
        atr_period = params.get("atr_period", 14)
        atr_mult = params.get("atr_multiplier", 1.5)

        if len(df) < max(lookback, atr_period) + 1:
            return Signal(action="hold")

        atr_series = calc_atr(df, atr_period)
        curr_atr = atr_series.iloc[-1]

        # Range measured from candles BEFORE current
        prior = df.iloc[-(lookback + 1) : -1]
        range_high = prior["high"].max()
        range_low = prior["low"].min()

        curr_close = df["close"].iloc[-1]
        buy_level = range_high + atr_mult * curr_atr
        sell_level = range_low - atr_mult * curr_atr

        if curr_close > buy_level:
            confidence = min((curr_close - buy_level) / curr_atr, 1.0)
            return Signal(
                action="buy",
                confidence=confidence,
                meta={
                    "range_high": range_high,
                    "buy_level": buy_level,
                    "close": curr_close,
                    "atr": curr_atr,
                },
            )
        if curr_close < sell_level:
            confidence = min((sell_level - curr_close) / curr_atr, 1.0)
            return Signal(
                action="sell",
                confidence=confidence,
                meta={
                    "range_low": range_low,
                    "sell_level": sell_level,
                    "close": curr_close,
                    "atr": curr_atr,
                },
            )
        return Signal(action="hold")
