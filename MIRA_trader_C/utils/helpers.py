"""
MIRA_trader_C – General utility helpers.
"""
from __future__ import annotations

from datetime import datetime, timezone


def now_utc() -> datetime:
    """Return the current UTC datetime (timezone-aware)."""
    return datetime.now(tz=timezone.utc)


def pct_change(old: float, new: float) -> float:
    """Return percentage change from old to new."""
    if old == 0:
        return 0.0
    return (new - old) / abs(old) * 100


def clamp(value: float, low: float, high: float) -> float:
    """Clamp value between low and high."""
    return max(low, min(high, value))


def round_to_precision(value: float, precision: int = 6) -> float:
    """Round a float to the given number of decimal places."""
    return round(value, precision)
