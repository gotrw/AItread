"""
MIRA_trader_C – Volatility Strategy.

Designed for choppy/squeeze market conditions where trend-following and
mean-reversion strategies underperform.  Detects Bollinger Band squeezes
(low volatility periods) and trades the subsequent expansion.

Entry logic
-----------
- LONG  when BB is squeezing (width below threshold) AND price breaks above
  upper band on an ATR expansion candle.
- SHORT when BB is squeezing AND price breaks below lower band on an ATR
  expansion candle.
"""
from __future__ import annotations

import pandas as pd

from indicators.technical import atr as calc_atr, bollinger_bands, sma
from strategies.base import BaseStrategy, Signal


class VolatilityStrategy(BaseStrategy):
    """Bollinger Band squeeze + ATR expansion breakout."""

    name = "volatility"

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        params = self.params
        bb_period: int = params.get("bb_period", 20)
        bb_std: float = params.get("bb_std", 2.0)
        atr_period: int = params.get("atr_period", 14)
        squeeze_threshold: float = params.get("squeeze_threshold", 0.02)
        expansion_mult: float = params.get("expansion_multiplier", 1.5)

        min_rows = max(bb_period, atr_period) + 5
        if len(df) < min_rows:
            return Signal(action="hold", strategy_name=self.name)

        close = df["close"]
        bb = bollinger_bands(close, bb_period, bb_std)
        atr_series = calc_atr(df, atr_period)

        curr_close = float(close.iloc[-1])
        curr_width = float(bb["width"].iloc[-1])
        curr_upper = float(bb["upper"].iloc[-1])
        curr_lower = float(bb["lower"].iloc[-1])
        curr_atr = float(atr_series.iloc[-1])

        # Average ATR over the lookback for expansion detection
        avg_atr = float(atr_series.iloc[-bb_period:].mean())

        # Squeeze: BB width is compressed
        in_squeeze = curr_width < squeeze_threshold

        # Expansion: current ATR significantly exceeds recent average
        atr_expanding = curr_atr > avg_atr * expansion_mult

        if in_squeeze and atr_expanding:
            if curr_close > curr_upper:
                confidence = min((curr_close - curr_upper) / curr_atr, 1.0)
                return Signal(
                    action="long",
                    confidence=confidence,
                    strategy_name=self.name,
                    meta={
                        "bb_width": curr_width,
                        "squeeze_threshold": squeeze_threshold,
                        "atr": curr_atr,
                        "avg_atr": avg_atr,
                        "close": curr_close,
                        "upper": curr_upper,
                    },
                )
            if curr_close < curr_lower:
                confidence = min((curr_lower - curr_close) / curr_atr, 1.0)
                return Signal(
                    action="short",
                    confidence=confidence,
                    strategy_name=self.name,
                    meta={
                        "bb_width": curr_width,
                        "squeeze_threshold": squeeze_threshold,
                        "atr": curr_atr,
                        "avg_atr": avg_atr,
                        "close": curr_close,
                        "lower": curr_lower,
                    },
                )

        return Signal(action="hold", strategy_name=self.name)
