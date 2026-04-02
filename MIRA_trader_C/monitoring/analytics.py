"""
MIRA_trader_C – Advanced Performance Analytics.

Computes risk-adjusted return metrics from a list of TradeRecord objects.
All functions are pure (no side effects) and operate on the trade history
stored in MetricsTracker.

Metrics
-------
- Sharpe Ratio      : mean(returns) / std(returns) * sqrt(periods_per_year)
- Sortino Ratio     : mean(returns) / downside_std * sqrt(periods_per_year)
- Calmar Ratio      : annualised return / max_drawdown
- Profit Factor     : gross_wins / gross_losses
- Win Rate          : wins / total_trades
- Avg Win / Avg Loss
- Max Consecutive Wins / Losses
- Recovery Factor   : total_pnl / max_drawdown_abs
- Expectancy        : (win_rate * avg_win) - (loss_rate * avg_loss)

Usage
-----
    from monitoring.analytics import compute_analytics
    from monitoring.metrics import MetricsTracker

    tracker = MetricsTracker(...)
    analytics = compute_analytics(tracker.get_metrics(), timeframe='1h')
    print(analytics)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

from monitoring.metrics import Metrics, TradeRecord

# Approximate trading periods per year for each timeframe
_PERIODS_PER_YEAR: Dict[str, float] = {
    "1m": 525_600,
    "5m": 105_120,
    "15m": 35_040,
    "30m": 17_520,
    "1h": 8_760,
    "2h": 4_380,
    "4h": 2_190,
    "6h": 1_460,
    "8h": 1_095,
    "12h": 730,
    "1d": 365,
    "1w": 52,
}


def compute_analytics(
    metrics: Metrics,
    timeframe: str = "1h",
    risk_free_rate: float = 0.0,
) -> Dict[str, object]:
    """Compute advanced analytics from recorded trade history.

    Args:
        metrics: Metrics object from MetricsTracker.
        timeframe: Candle timeframe for annualisation (default '1h').
        risk_free_rate: Annual risk-free rate for Sharpe calculation (default 0.0).

    Returns:
        Dictionary of metric names to values.
    """
    trades = metrics.trade_history
    if not trades:
        return _empty_analytics()

    closed = [t for t in trades if t.exit_price is not None]
    if not closed:
        return _empty_analytics()

    returns = [t.pnl_pct for t in closed]
    periods = _PERIODS_PER_YEAR.get(timeframe, 8_760)
    ann_factor = math.sqrt(periods)

    # ── Basic stats ────────────────────────────────────────────
    n = len(returns)
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]
    win_rate = len(wins) / n if n else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0

    # ── Profit factor ──────────────────────────────────────────
    gross_win = sum(t.pnl for t in closed if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in closed if t.pnl <= 0))
    profit_factor = gross_win / gross_loss if gross_loss else float("inf")

    # ── Expectancy ─────────────────────────────────────────────
    loss_rate = 1.0 - win_rate
    expectancy = win_rate * avg_win - loss_rate * avg_loss

    # ── Sharpe Ratio ───────────────────────────────────────────
    mean_ret = sum(returns) / n
    variance = sum((r - mean_ret) ** 2 for r in returns) / max(n - 1, 1)
    std_ret = math.sqrt(variance)
    rf_per_period = risk_free_rate / periods if periods > 0 else 0.0
    sharpe = ((mean_ret - rf_per_period) / std_ret * ann_factor) if std_ret > 0 else 0.0

    # ── Sortino Ratio ──────────────────────────────────────────
    downside = [r for r in returns if r < rf_per_period]
    if downside:
        down_var = sum((r - rf_per_period) ** 2 for r in downside) / max(len(downside) - 1, 1)
        down_std = math.sqrt(down_var)
        sortino = ((mean_ret - rf_per_period) / down_std * ann_factor) if down_std > 0 else 0.0
    else:
        sortino = float("inf")

    # ── Max Drawdown ───────────────────────────────────────────
    cum_pnl = 0.0
    peak = 0.0
    max_dd_pct = 0.0
    max_dd_abs = 0.0
    for t in closed:
        cum_pnl += t.pnl
        if cum_pnl > peak:
            peak = cum_pnl
        dd = peak - cum_pnl
        dd_pct = (dd / peak * 100) if peak > 0 else 0.0
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
            max_dd_abs = dd

    # ── Calmar Ratio ───────────────────────────────────────────
    total_pnl_pct = sum(returns)
    # Annualise: scale by (periods / n trades) – rough approximation
    ann_return = total_pnl_pct * (periods / n) if n > 0 else 0.0
    calmar = (ann_return / max_dd_pct) if max_dd_pct > 0 else float("inf")

    # ── Recovery Factor ────────────────────────────────────────
    total_pnl = sum(t.pnl for t in closed)
    recovery_factor = (total_pnl / max_dd_abs) if max_dd_abs > 0 else float("inf")

    # ── Consecutive wins / losses ──────────────────────────────
    max_consec_wins, max_consec_losses = _max_consecutive(returns)

    return {
        "total_trades": n,
        "win_rate": f"{win_rate:.1%}",
        "profit_factor": f"{profit_factor:.2f}",
        "expectancy_pct": f"{expectancy:.3f}%",
        "avg_win_pct": f"{avg_win:.3f}%",
        "avg_loss_pct": f"{avg_loss:.3f}%",
        "sharpe_ratio": f"{sharpe:.2f}",
        "sortino_ratio": f"{sortino:.2f}" if sortino != float("inf") else "∞",
        "calmar_ratio": f"{calmar:.2f}" if calmar != float("inf") else "∞",
        "max_drawdown_pct": f"{max_dd_pct:.2f}%",
        "recovery_factor": f"{recovery_factor:.2f}" if recovery_factor != float("inf") else "∞",
        "max_consecutive_wins": max_consec_wins,
        "max_consecutive_losses": max_consec_losses,
        "total_pnl": f"${total_pnl:.2f}",
        "timeframe": timeframe,
    }


def compute_per_symbol_analytics(
    metrics: Metrics,
    timeframe: str = "1h",
) -> Dict[str, Dict[str, object]]:
    """Compute analytics broken down per symbol.

    Returns a dict mapping symbol → analytics dict.
    """
    from collections import defaultdict
    from dataclasses import replace

    symbol_trades: Dict[str, List[TradeRecord]] = defaultdict(list)
    for t in metrics.trade_history:
        symbol_trades[t.symbol].append(t)

    result: Dict[str, Dict[str, object]] = {}
    for symbol, trades in symbol_trades.items():
        fake_metrics = Metrics(trade_history=trades)
        fake_metrics.total_trades = len(trades)
        for t in trades:
            fake_metrics.total_pnl += t.pnl
            if t.pnl > 0:
                fake_metrics.winning_trades += 1
            else:
                fake_metrics.losing_trades += 1
        result[symbol] = compute_analytics(fake_metrics, timeframe)

    return result


def _max_consecutive(returns: List[float]) -> tuple[int, int]:
    """Return (max_consecutive_wins, max_consecutive_losses)."""
    max_wins = max_losses = curr_wins = curr_losses = 0
    for r in returns:
        if r > 0:
            curr_wins += 1
            curr_losses = 0
            max_wins = max(max_wins, curr_wins)
        else:
            curr_losses += 1
            curr_wins = 0
            max_losses = max(max_losses, curr_losses)
    return max_wins, max_losses


def _empty_analytics() -> Dict[str, object]:
    return {
        "total_trades": 0,
        "win_rate": "0.0%",
        "profit_factor": "0.00",
        "expectancy_pct": "0.000%",
        "avg_win_pct": "0.000%",
        "avg_loss_pct": "0.000%",
        "sharpe_ratio": "0.00",
        "sortino_ratio": "0.00",
        "calmar_ratio": "0.00",
        "max_drawdown_pct": "0.00%",
        "recovery_factor": "0.00",
        "max_consecutive_wins": 0,
        "max_consecutive_losses": 0,
        "total_pnl": "$0.00",
        "timeframe": "1h",
    }
