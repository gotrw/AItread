"""Tests for signal_scanner/scanner.py."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from signal_scanner.scanner import ScannedSignal, SignalScanner, get_recent_signals, _clear_signals


# ── ScannedSignal ────────────────────────────────────────────

def test_scanned_signal_creation():
    sig = ScannedSignal(
        symbol="BTC/USDT", action="long", confidence=0.75,
        strategy="trend_follow", regime="trending", price=30000.0,
    )
    assert sig.symbol == "BTC/USDT"
    assert sig.action == "long"
    assert sig.confidence == pytest.approx(0.75)
    assert sig.regime == "trending"
    assert sig.price == pytest.approx(30000.0)
    assert sig.timestamp  # auto-filled


def test_scanned_signal_to_dict():
    sig = ScannedSignal(symbol="ETH/USDT", action="hold", confidence=0.0, strategy="multi_strategy")
    d = sig.to_dict()
    assert d["symbol"] == "ETH/USDT"
    assert d["action"] == "hold"
    assert "timestamp" in d
    assert "risk_label" in d


def test_risk_label_low():
    sig = ScannedSignal(symbol="X", action="long", confidence=0.3, strategy="s")
    assert sig.risk_label() == "LOW"


def test_risk_label_med():
    sig = ScannedSignal(symbol="X", action="long", confidence=0.55, strategy="s")
    assert sig.risk_label() == "MED"

def test_risk_label_high():
    sig = ScannedSignal(symbol="X", action="long", confidence=0.8, strategy="s")
    assert sig.risk_label() == "HIGH"


def test_risk_label_hold_returns_empty():
    sig = ScannedSignal(symbol="X", action="hold", confidence=0.9, strategy="s")
    assert sig.risk_label() == ""


# ── SignalScanner.scan_once ──────────────────────────────────

class FakeProvider:
    """Data provider that returns synthetic OHLCV without network."""
    def fetch_ohlcv(self, symbol, timeframe, limit=500):
        np.random.seed(hash(symbol) % (2**31))
        prices = list(30000 + np.cumsum(np.random.randn(200) * 100))
        close = np.array(prices)
        return pd.DataFrame(
            {
                "open": close * 0.999,
                "high": close * 1.005,
                "low": close * 0.995,
                "close": close,
                "volume": np.ones(200) * 1000,
            },
            index=pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC"),
        )


class FakeStrategy:
    """Strategy that always returns 'hold'."""
    name = "fake"

    def generate_signal(self, df):
        from strategies.base import Signal
        return Signal(action="hold", confidence=0.0)


def _make_cfg(symbols=None):
    from config import load_config
    cfg = load_config()
    if symbols is not None:
        cfg.setdefault("data", {})["symbols"] = symbols
    cfg.setdefault("scanner", {})["interval_seconds"] = 3600
    return cfg


def test_scan_once_returns_list():
    _clear_signals()
    cfg = _make_cfg(["BTC/USDT", "ETH/USDT"])
    scanner = SignalScanner(cfg, FakeProvider(), FakeStrategy())
    signals = scanner.scan_once()
    assert isinstance(signals, list)
    assert len(signals) == 2


def test_scan_once_signals_have_required_fields():
    _clear_signals()
    cfg = _make_cfg(["BTC/USDT"])
    scanner = SignalScanner(cfg, FakeProvider(), FakeStrategy())
    signals = scanner.scan_once()
    for sig in signals:
        assert hasattr(sig, "symbol")
        assert hasattr(sig, "action")
        assert hasattr(sig, "confidence")
        assert hasattr(sig, "timestamp")


def test_scan_once_stored_in_process():
    _clear_signals()
    cfg = _make_cfg(["BTC/USDT"])
    scanner = SignalScanner(cfg, FakeProvider(), FakeStrategy())
    scanner.scan_once()
    # All signals are stored (including "hold") so in-process store should have at least 1 entry
    recent = get_recent_signals(10)
    assert len(recent) >= 1
    found = [s for s in recent if s["symbol"] == "BTC/USDT"]
    assert len(found) >= 1


def test_get_recent_signals_limit():
    _clear_signals()
    cfg = _make_cfg(["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    scanner = SignalScanner(cfg, FakeProvider(), FakeStrategy())
    scanner.scan_once()
    scanner.scan_once()
    recent = get_recent_signals(2)
    assert len(recent) <= 2


def test_scan_once_error_on_bad_data_skips():
    """Scanner should not raise even if data provider fails for a symbol."""
    _clear_signals()

    class FailProvider:
        def fetch_ohlcv(self, symbol, tf, limit=500):
            raise RuntimeError("Network error")

    cfg = _make_cfg(["BTC/USDT"])
    scanner = SignalScanner(cfg, FailProvider(), FakeStrategy())
    # Should not raise
    signals = scanner.scan_once()
    assert isinstance(signals, list)
