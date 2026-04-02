"""
Tests for MIRA_trader_C backtesting engine.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from backtesting.engine import BacktestEngine, BacktestResults
from strategies.base import BaseStrategy, Signal


class AlwaysBuyStrategy(BaseStrategy):
    """Test strategy that always signals BUY."""
    name = "always_buy"

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        return Signal(action="buy", confidence=1.0)


class AlwaysHoldStrategy(BaseStrategy):
    """Test strategy that always signals HOLD."""
    name = "always_hold"

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        return Signal(action="hold")


@pytest.fixture
def minimal_cfg():
    return {
        "backtest": {"fee_rate": 0.001, "slippage_pct": 0.05},
        "risk": {
            "initial_capital": 1000.0,
            "max_risk_per_trade_pct": 1.0,
            "max_daily_loss_pct": 100.0,
            "max_open_positions": 3,
            "max_exposure_per_symbol_pct": 100,
            "stop_loss_pct": 1.5,
            "take_profit_pct": 3.0,
            "cooldown_after_loss_trades": 0,
            "kill_switch": False,
        },
        "strategy": {
            "trend_follow": {"slow_ema": 21},
            "mean_reversion": {"bb_period": 20},
            "breakout": {"lookback": 20},
        },
    }


@pytest.fixture
def synthetic_df():
    """200-bar uptrending OHLCV DataFrame."""
    np.random.seed(42)
    close = 30000 + np.cumsum(np.random.randn(200) * 50 + 5)
    idx = pd.date_range("2024-01-01", periods=200, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": np.random.uniform(100, 500, 200),
        },
        index=idx,
    )


# ── BacktestResults ───────────────────────────────────────────

def test_results_summary_no_trades():
    res = BacktestResults()
    summary = res.summary()
    assert "error" in summary


def test_results_summary_with_trades():
    from backtesting.engine import Trade
    import pandas as pd

    trade = Trade(
        symbol="BTC/USDT",
        entry_time=pd.Timestamp("2024-01-01", tz="UTC"),
        entry_price=30000.0,
        qty=0.001,
        stop_loss=29550.0,
        take_profit=30900.0,
        exit_time=pd.Timestamp("2024-01-02", tz="UTC"),
        exit_price=30900.0,
        pnl=0.9,
        pnl_pct=0.09,
        exit_reason="take_profit",
    )
    equity = pd.Series([1000.0, 1000.9])
    res = BacktestResults(trades=[trade], equity_curve=equity, initial_capital=1000.0)
    summary = res.summary()
    assert "win_rate" in summary
    assert "profit_factor" in summary
    assert "total_trades" in summary


# ── BacktestEngine ────────────────────────────────────────────

def test_engine_hold_strategy_no_trades(minimal_cfg, synthetic_df):
    strategy = AlwaysHoldStrategy({})
    engine = BacktestEngine(minimal_cfg, strategy)
    results = engine.run(synthetic_df, "BTC/USDT")
    assert len(results.trades) == 0


def test_engine_buy_strategy_opens_trades(minimal_cfg, synthetic_df):
    strategy = AlwaysBuyStrategy({})
    engine = BacktestEngine(minimal_cfg, strategy)
    results = engine.run(synthetic_df, "BTC/USDT")
    assert len(results.trades) >= 1


def test_engine_equity_curve_not_empty(minimal_cfg, synthetic_df):
    strategy = AlwaysBuyStrategy({})
    engine = BacktestEngine(minimal_cfg, strategy)
    results = engine.run(synthetic_df, "BTC/USDT")
    assert len(results.equity_curve) > 0


def test_engine_summary_keys(minimal_cfg, synthetic_df):
    strategy = AlwaysBuyStrategy({})
    engine = BacktestEngine(minimal_cfg, strategy)
    results = engine.run(synthetic_df, "BTC/USDT")
    summary = results.summary()
    expected_keys = {"total_trades", "win_rate", "profit_factor", "total_pnl", "max_drawdown"}
    assert expected_keys.issubset(summary.keys())
