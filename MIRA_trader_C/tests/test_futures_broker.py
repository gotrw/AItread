"""Tests for PaperFuturesBroker."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from execution.futures_broker import PaperFuturesBroker


def make_broker(capital: float = 10_000.0) -> PaperFuturesBroker:
    return PaperFuturesBroker({"USDT": capital})


# ── Order placement ──────────────────────────────────────────

def test_place_buy_order_returns_order():
    broker = make_broker()
    order = broker.place_market_order("BTC/USDT", "buy", 0.01)
    assert order is not None
    assert order.symbol == "BTC/USDT"
    assert order.side == "buy"
    assert order.qty == 0.01


def test_fill_long_creates_position():
    broker = make_broker()
    order = broker.place_market_order("BTC/USDT", "buy", 0.01)
    broker.fill_paper_order(order, 30_000.0, leverage=5)
    positions = broker.get_positions()
    assert "BTC/USDT" in positions
    pos = positions["BTC/USDT"]
    assert pos["side"] == "buy"
    assert pos["qty"] == pytest.approx(0.01)
    assert pos["entry_price"] == pytest.approx(30_000.0)
    assert pos["leverage"] == 5


def test_liquidation_price_long():
    """Isolated-margin long liquidation price should be below entry."""
    broker = make_broker()
    order = broker.place_market_order("BTC/USDT", "buy", 0.01)
    broker.fill_paper_order(order, 30_000.0, leverage=10)
    pos = broker.get_positions()["BTC/USDT"]
    liq = pos["liquidation_price"]
    assert liq < 30_000.0, "Long liquidation price must be below entry"


def test_liquidation_price_short():
    """Isolated-margin short liquidation price should be above entry."""
    broker = make_broker()
    order = broker.place_market_order("BTC/USDT", "sell", 0.01)
    broker.fill_paper_order(order, 30_000.0, leverage=10)
    pos = broker.get_positions()["BTC/USDT"]
    liq = pos["liquidation_price"]
    assert liq > 30_000.0, "Short liquidation price must be above entry"


def test_is_liquidated_long_at_liq_price():
    broker = make_broker()
    order = broker.place_market_order("BTC/USDT", "buy", 0.01)
    broker.fill_paper_order(order, 30_000.0, leverage=10)
    liq = broker.get_positions()["BTC/USDT"]["liquidation_price"]
    assert broker.is_liquidated("BTC/USDT", liq - 1)


def test_not_liquidated_long_above_entry():
    broker = make_broker()
    order = broker.place_market_order("BTC/USDT", "buy", 0.01)
    broker.fill_paper_order(order, 30_000.0, leverage=5)
    assert not broker.is_liquidated("BTC/USDT", 31_000.0)


def test_unrealised_pnl_long_gain():
    broker = make_broker()
    order = broker.place_market_order("BTC/USDT", "buy", 0.01)
    broker.fill_paper_order(order, 30_000.0, leverage=5)
    pnl = broker.update_pnl("BTC/USDT", 31_000.0)
    # price gain 1000 × qty 0.01 × leverage 5 = 50
    assert pnl == pytest.approx(50.0)


def test_unrealised_pnl_long_loss():
    broker = make_broker()
    order = broker.place_market_order("BTC/USDT", "buy", 0.01)
    broker.fill_paper_order(order, 30_000.0, leverage=5)
    pnl = broker.update_pnl("BTC/USDT", 29_000.0)
    assert pnl == pytest.approx(-50.0)


def test_unrealised_pnl_short_gain():
    broker = make_broker()
    order = broker.place_market_order("BTC/USDT", "sell", 0.01)
    broker.fill_paper_order(order, 30_000.0, leverage=5)
    pnl = broker.update_pnl("BTC/USDT", 29_000.0)
    # short gains when price drops
    assert pnl == pytest.approx(50.0)


def test_close_position_removes_it():
    broker = make_broker()
    order = broker.place_market_order("BTC/USDT", "buy", 0.01)
    broker.fill_paper_order(order, 30_000.0, leverage=5)
    closed = broker.close_position("BTC/USDT")
    assert closed is not None
    assert "BTC/USDT" not in broker.get_positions()


def test_is_liquidated_unknown_symbol_returns_false():
    broker = make_broker()
    assert not broker.is_liquidated("UNKNOWN/USDT", 30_000.0)


def test_get_positions_empty_initially():
    broker = make_broker()
    assert broker.get_positions() == {}
