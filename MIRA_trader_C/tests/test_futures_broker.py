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


# ── LiveBroker / FuturesBroker mocked tests ─────────────────

class _MockExchange:
    """Minimal mock that mimics the CCXT exchange interface."""

    def __init__(self) -> None:
        self._orders: dict[str, dict] = {}
        self._order_counter = 0

    def fetch_balance(self) -> dict:
        return {"free": {"USDT": 5000.0, "BTC": 0.0}, "total": {"USDT": 5000.0}}

    def create_market_order(self, symbol, side, qty, params=None):
        self._order_counter += 1
        oid = str(self._order_counter)
        self._orders[oid] = {
            "id": oid, "symbol": symbol, "side": side, "amount": qty,
            "filled": qty, "average": 30000.0, "price": 30000.0, "status": "closed",
        }
        return self._orders[oid]

    def create_order(self, symbol, order_type, side, qty, params=None):
        self._order_counter += 1
        oid = str(self._order_counter)
        stop_price = (params or {}).get("stopPrice", 0.0)
        self._orders[oid] = {
            "id": oid, "symbol": symbol, "type": order_type,
            "side": side, "amount": qty, "stopPrice": stop_price, "status": "open",
        }
        return self._orders[oid]

    def cancel_order(self, order_id, symbol):
        if order_id in self._orders:
            self._orders[order_id]["status"] = "cancelled"
        return {"id": order_id, "status": "cancelled"}

    def fetch_open_orders(self, symbol=None):
        return [o for o in self._orders.values() if o["status"] == "open"]

    def set_leverage(self, leverage, symbol):
        return {}

    def set_margin_mode(self, mode, symbol):
        return {}

    def set_sandbox_mode(self, enabled):
        pass


def _make_live_futures_broker() -> "FuturesBroker":
    """Create a FuturesBroker with a mocked exchange (no real API calls)."""
    from execution.futures_broker import FuturesBroker
    broker = FuturesBroker.__new__(FuturesBroker)
    broker._exchange = _MockExchange()
    broker._futures_cfg = {}
    return broker


def test_futures_broker_get_balance_uses_free():
    """get_balance() must return free balances, not totals."""
    broker = _make_live_futures_broker()
    bal = broker.get_balance()
    assert bal == {"USDT": 5000.0}


def test_futures_broker_place_bracket_orders_returns_ids():
    broker = _make_live_futures_broker()
    result = broker.place_bracket_orders("BTC/USDT", "long", 0.01, 29000.0, 31000.0)
    assert "sl_order_id" in result
    assert "tp_order_id" in result
    assert result["sl_order_id"] != ""
    assert result["tp_order_id"] != ""


def test_futures_broker_bracket_short_uses_buy_side():
    broker = _make_live_futures_broker()
    mock: _MockExchange = broker._exchange  # type: ignore[assignment]
    broker.place_bracket_orders("ETH/USDT", "short", 0.1, 3500.0, 2500.0)
    # Both bracket orders for a short must be 'buy' (to close the short)
    bracket_orders = [o for o in mock._orders.values() if o["symbol"] == "ETH/USDT"]
    assert all(o["side"] == "buy" for o in bracket_orders)


def test_futures_broker_cancel_bracket_orders():
    broker = _make_live_futures_broker()
    mock: _MockExchange = broker._exchange  # type: ignore[assignment]
    result = broker.place_bracket_orders("BTC/USDT", "long", 0.01, 29000.0, 31000.0)
    sl_id, tp_id = result["sl_order_id"], result["tp_order_id"]

    broker.cancel_bracket_orders("BTC/USDT", sl_id, tp_id)

    assert mock._orders[sl_id]["status"] == "cancelled"
    assert mock._orders[tp_id]["status"] == "cancelled"


def test_futures_broker_cancel_bracket_orders_empty_ids():
    """cancel_bracket_orders with empty IDs should not raise."""
    broker = _make_live_futures_broker()
    broker.cancel_bracket_orders("BTC/USDT", "", "")  # should not raise


def test_futures_broker_set_leverage_ok():
    broker = _make_live_futures_broker()
    assert broker.set_leverage("BTC/USDT", 10) is True


def test_futures_broker_set_margin_type_ok():
    broker = _make_live_futures_broker()
    assert broker.set_margin_type("BTC/USDT", "isolated") is True


def test_live_broker_get_balance_bug_fix():
    """Regression: get_balance() must not call info['free'] on a float."""
    from execution.live_broker import LiveBroker
    broker = LiveBroker.__new__(LiveBroker)
    broker._exchange = _MockExchange()
    bal = broker.get_balance()
    # BTC is 0.0 so excluded; USDT 5000 is included
    assert bal == {"USDT": 5000.0}
    assert "BTC" not in bal
