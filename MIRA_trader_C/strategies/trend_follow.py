"""
MIRA_trader_C – Trend-Following Strategy.

Uses dual EMA crossover (fast/slow) combined with RSI to confirm momentum.
- BUY  when fast EMA crosses above slow EMA AND RSI > buy_threshold
- SELL when fast EMA crosses below slow EMA AND RSI < sell_threshold
"""
from __future__ import annotations

import pandas as pd

from indicators.technical import ema, rsi as calc_rsi
from strategies.base import BaseStrategy, Signal


class TrendFollowStrategy(BaseStrategy):
    """Dual EMA crossover + RSI confirmation."""

    name = "trend_follow"

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        params = self.params
        fast = params.get("fast_ema", 9)
        slow = params.get("slow_ema", 21)
        rsi_period = params.get("rsi_period", 14)
        rsi_buy = params.get("rsi_buy_threshold", 50)
        rsi_sell = params.get("rsi_sell_threshold", 50)

        close = df["close"]
        fast_ema = ema(close, fast)
        slow_ema = ema(close, slow)
        rsi_series = calc_rsi(close, rsi_period)

        # Current and previous bars
        curr_fast, prev_fast = fast_ema.iloc[-1], fast_ema.iloc[-2]
        curr_slow, prev_slow = slow_ema.iloc[-1], slow_ema.iloc[-2]
        curr_rsi = rsi_series.iloc[-1]

        cross_up = prev_fast <= prev_slow and curr_fast > curr_slow
        cross_down = prev_fast >= prev_slow and curr_fast < curr_slow

        if cross_up and curr_rsi > rsi_buy:
            return Signal(
                action="buy",
                confidence=min((curr_rsi - rsi_buy) / 50, 1.0),
                meta={"rsi": curr_rsi, "fast_ema": curr_fast, "slow_ema": curr_slow},
            )
        if cross_down and curr_rsi < rsi_sell:
            return Signal(
                action="sell",
                confidence=min((rsi_sell - curr_rsi) / 50, 1.0),
                meta={"rsi": curr_rsi, "fast_ema": curr_fast, "slow_ema": curr_slow},
            )
        return Signal(action="hold")
