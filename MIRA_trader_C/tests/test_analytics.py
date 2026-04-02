"""Tests for monitoring/analytics.py."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from monitoring.analytics import compute_analytics, compute_per_symbol_analytics, _empty_analytics
from monitoring.metrics import Metrics, TradeRecord


def _make_metrics(trades: list[tuple]) -> Metrics:
    """Build a Metrics object from (symbol, side, pnl) tuples."""
    m = Metrics()
    for symbol, side, pnl in trades:
        entry = 30000.0
        exit_p = entry + pnl / 0.001  # qty = 0.001
        rec = TradeRecord(
            symbol=symbol, side=side,
            entry_price=entry, exit_price=exit_p,
            qty=0.001, pnl=pnl, pnl_pct=pnl / 30.0,
            timestamp="2024-01-01T00:00:00", exit_reason="take_profit",
        )
        m.trade_history.append(rec)
        m.total_trades += 1
        m.total_pnl += pnl
        if pnl > 0:
            m.winning_trades += 1
        else:
            m.losing_trades += 1
    return m


# ── _empty_analytics ─────────────────────────────────────────

def test_empty_analytics_keys():
    result = _empty_analytics()
    assert "sharpe_ratio" in result
    assert "sortino_ratio" in result
    assert "win_rate" in result
    assert "profit_factor" in result


def test_empty_analytics_no_trades():
    m = Metrics()
    result = compute_analytics(m)
    # Should return _empty_analytics() which has 0.0 for numeric fields
    assert result["sharpe_ratio"] == 0.0 or result["sharpe_ratio"] == "0.00"
    assert result["sortino_ratio"] == 0.0 or result["sortino_ratio"] == "0.00"


# ── compute_analytics ────────────────────────────────────────

def test_analytics_positive_pnl():
    m = _make_metrics([
        ("BTC/USDT", "long", 10.0),
        ("BTC/USDT", "long", 8.0),
        ("BTC/USDT", "long", -3.0),
        ("BTC/USDT", "long", 12.0),
    ])
    result = compute_analytics(m)
    assert result["win_rate"] is not None
    assert "66.7%" in result["win_rate"] or "75.0%" in result["win_rate"]


def test_profit_factor_greater_than_1_for_profitable():
    m = _make_metrics([
        ("BTC/USDT", "long", 20.0),
        ("BTC/USDT", "long", 15.0),
        ("BTC/USDT", "long", -5.0),
    ])
    result = compute_analytics(m)
    pf = float(result["profit_factor"])
    assert pf > 1.0


def test_profit_factor_less_than_1_for_losing():
    m = _make_metrics([
        ("BTC/USDT", "long", -20.0),
        ("BTC/USDT", "long", -15.0),
        ("BTC/USDT", "long", 5.0),
    ])
    result = compute_analytics(m)
    pf = float(result["profit_factor"])
    assert pf < 1.0


def test_sharpe_positive_for_profitable_stream():
    m = _make_metrics([
        ("BTC/USDT", "long", 10.0),
        ("BTC/USDT", "long", 12.0),
        ("BTC/USDT", "long", 9.0),
        ("BTC/USDT", "long", 11.0),
        ("BTC/USDT", "long", 10.5),
    ])
    result = compute_analytics(m)
    sharpe = float(result["sharpe_ratio"])
    assert sharpe > 0


def _pct_to_float(s) -> float:
    """Convert a percentage string like '12.34%' to 12.34, or return float directly."""
    if isinstance(s, (int, float)):
        return float(s)
    return float(str(s).replace('%', '').strip())


def test_max_drawdown_zero_for_only_wins():
    m = _make_metrics([
        ("BTC/USDT", "long", 10.0),
        ("BTC/USDT", "long", 5.0),
        ("BTC/USDT", "long", 8.0),
    ])
    result = compute_analytics(m)
    assert _pct_to_float(result["max_drawdown_pct"]) == pytest.approx(0.0)


def test_max_drawdown_positive_for_loss():
    m = _make_metrics([
        ("BTC/USDT", "long", 10.0),
        ("BTC/USDT", "long", -20.0),
        ("BTC/USDT", "long", 5.0),
    ])
    result = compute_analytics(m)
    assert _pct_to_float(result["max_drawdown_pct"]) > 0


# ── compute_per_symbol_analytics ─────────────────────────────

def test_per_symbol_returns_all_symbols():
    m = _make_metrics([
        ("BTC/USDT", "long", 10.0),
        ("ETH/USDT", "long", -5.0),
        ("BTC/USDT", "short", 7.0),
    ])
    result = compute_per_symbol_analytics(m)
    assert "BTC/USDT" in result
    assert "ETH/USDT" in result


def test_per_symbol_btc_has_two_trades():
    m = _make_metrics([
        ("BTC/USDT", "long", 10.0),
        ("ETH/USDT", "long", -5.0),
        ("BTC/USDT", "short", 7.0),
    ])
    result = compute_per_symbol_analytics(m)
    btc = result["BTC/USDT"]
    assert int(btc.get("total_trades", 0)) == 2
