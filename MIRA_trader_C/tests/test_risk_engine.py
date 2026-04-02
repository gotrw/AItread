"""
Tests for MIRA_trader_C risk engine.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from strategies.base import Signal
from risk.engine import RiskEngine


@pytest.fixture
def base_cfg():
    return {
        "max_risk_per_trade_pct": 1.0,
        "max_daily_loss_pct": 3.0,
        "max_open_positions": 3,
        "max_exposure_per_symbol_pct": 30,
        "stop_loss_pct": 1.5,
        "take_profit_pct": 3.0,
        "cooldown_after_loss_trades": 2,
        "kill_switch": False,
    }


@pytest.fixture
def engine(base_cfg):
    return RiskEngine(base_cfg, initial_capital=1000.0)


def make_buy_signal():
    return Signal(action="buy", confidence=0.8)


# ── approve_trade ─────────────────────────────────────────────

def test_approve_normal_trade(engine):
    approved, reason = engine.approve_trade("BTC/USDT", make_buy_signal(), 50000.0, {})
    assert approved, reason


def test_reject_kill_switch(base_cfg):
    base_cfg["kill_switch"] = True
    eng = RiskEngine(base_cfg, 1000.0)
    approved, reason = eng.approve_trade("BTC/USDT", make_buy_signal(), 50000.0, {})
    assert not approved
    assert "kill_switch" in reason


def test_reject_hold_signal(engine):
    sig = Signal(action="hold")
    approved, reason = engine.approve_trade("BTC/USDT", sig, 50000.0, {})
    assert not approved
    assert "hold" in reason


def test_reject_daily_loss_limit(engine):
    # Simulate reaching daily loss limit
    engine.record_trade_result(-3.0)
    approved, reason = engine.approve_trade("BTC/USDT", make_buy_signal(), 50000.0, {})
    assert not approved
    assert "daily loss" in reason


def test_reject_max_open_positions(base_cfg):
    base_cfg["max_open_positions"] = 2
    eng = RiskEngine(base_cfg, 1000.0)
    open_positions = {
        "BTC/USDT": {"value": 200},
        "ETH/USDT": {"value": 200},
    }
    approved, reason = eng.approve_trade("SOL/USDT", make_buy_signal(), 100.0, open_positions)
    assert not approved
    assert "max open positions" in reason


def test_cooldown_after_loss(engine):
    engine.record_trade_result(-1.0)
    # First trade after loss should be rejected (cooldown=2)
    approved, reason = engine.approve_trade("BTC/USDT", make_buy_signal(), 50000.0, {})
    assert not approved
    assert "cooldown" in reason


# ── compute_sizing ────────────────────────────────────────────

def test_sizing_returns_positive_qty(engine):
    qty, sl, tp = engine.compute_sizing(50000.0)
    assert qty > 0
    assert sl < 50000.0
    assert tp > 50000.0


def test_sizing_risk_amount(engine):
    price = 50000.0
    qty, sl, tp = engine.compute_sizing(price)
    # Risk = qty * (price - sl) should be ≈ 1% of capital = $10
    risk = qty * (price - sl)
    assert abs(risk - 10.0) < 0.5  # within $0.50 tolerance


def test_sizing_sl_tp_customisable(engine):
    qty, sl, tp = engine.compute_sizing(1000.0, sl_pct=2.0, tp_pct=6.0)
    assert abs(sl - 980.0) < 0.1
    assert abs(tp - 1060.0) < 0.1


# ── record_trade_result ───────────────────────────────────────

def test_capital_unchanged_by_record(engine):
    initial = engine.capital
    engine.record_trade_result(2.0)
    # capital is managed externally; record_trade_result only updates PnL tracking
    assert engine.capital == initial
