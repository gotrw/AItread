"""Tests for notifications/telegram.py."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from notifications.telegram import TelegramNotifier, build_notifier
from signal_scanner.scanner import ScannedSignal


# ── build_notifier factory ───────────────────────────────────

def test_build_notifier_disabled_returns_none():
    notifier = build_notifier({"notifications": {"telegram": {"enabled": False}}})
    assert notifier is None


def test_build_notifier_no_section_returns_none():
    notifier = build_notifier({})
    assert notifier is None


def test_build_notifier_enabled_without_token_creates_disabled_notifier():
    # enabled=True but no token → TelegramNotifier is created but immediately disabled
    notifier = build_notifier({"notifications": {"telegram": {"enabled": True, "bot_token": "", "chat_id": ""}}})
    # Either returns None or a disabled TelegramNotifier
    if notifier is not None:
        assert not notifier._enabled


def test_telegram_notifier_disabled_does_not_send():
    tn = TelegramNotifier({"enabled": False, "bot_token": "fake", "chat_id": "123"})
    # No exception should be raised, and _enabled must be False
    assert not tn._enabled
    # All send methods must be no-ops when disabled
    sig = ScannedSignal(symbol="X", action="long", confidence=0.7, strategy="s")
    tn.send_signal(sig)
    tn.send_order("X", "buy", 0.1, 30000.0)
    tn.send_risk_event("kill_switch", {"msg": "test"})
    tn.send_error("test error")
    tn.send_summary({"win_rate": "60%"})
    tn.send_system_status("Testing")


# ── TelegramNotifier internal helpers ────────────────────────

def test_format_signal_contains_symbol():
    tn = TelegramNotifier({"enabled": False})
    sig = ScannedSignal(symbol="BTC/USDT", action="long", confidence=0.85, strategy="trend_follow", regime="trending", price=30000.0)
    msg = tn._format_signal(sig)
    assert "BTC/USDT" in msg
    assert "LONG" in msg or "long" in msg

def test_format_order_contains_symbol_and_side():
    tn = TelegramNotifier({"enabled": False})
    msg = tn._format_order("BTC/USDT", "buy", 0.01, 30000.0)
    assert "BTC/USDT" in msg
    assert "buy" in msg.lower() or "BUY" in msg


def test_format_risk_event_contains_type():
    tn = TelegramNotifier({"enabled": False})
    msg = tn._format_risk_event("daily_loss_limit", {"symbol": "ETH/USDT", "pnl": -50.0})
    assert "daily_loss_limit" in msg or "daily" in msg.lower()


def test_batch_summary_ready_false_initially():
    tn = TelegramNotifier({"enabled": False, "batch_interval_hours": 24})
    assert not tn._batch_summary_ready()
