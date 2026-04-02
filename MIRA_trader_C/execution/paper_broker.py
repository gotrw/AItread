"""
MIRA_trader_C – Paper Broker.

Simulates order execution without placing real trades.
Uses the last known price as the fill price.
Safe to use for strategy development and risk-free testing.
"""
from __future__ import annotations

import logging
import uuid
from typing import Dict

from execution.base import BrokerBase, Order, OrderSide

logger = logging.getLogger(__name__)


class PaperBroker(BrokerBase):
    """In-memory simulated broker for paper trading."""

    def __init__(self, initial_balance: Dict[str, float] | None = None) -> None:
        self._balance: Dict[str, float] = initial_balance or {"USDT": 1000.0}
        self._open_orders: Dict[str, Order] = {}

    def place_market_order(self, symbol: str, side: OrderSide, qty: float) -> Order:
        """Fill at the last provided price (caller must pass price via qty heuristic)."""
        order_id = str(uuid.uuid4())[:8]
        # Price is determined by the caller; paper broker just records the fill.
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            qty=qty,
            price=0.0,
            status="open",
        )
        logger.info(
            "[PAPER] Market %s %s qty=%.6f → order_id=%s",
            side.upper(),
            symbol,
            qty,
            order_id,
        )
        return order

    def fill_paper_order(self, order: Order, fill_price: float) -> Order:
        """Simulate an immediate fill at `fill_price`."""
        order.filled_qty = order.qty
        order.filled_price = fill_price
        order.price = fill_price
        order.status = "filled"

        quote, base = symbol_parts(order.symbol)
        if order.side == "buy":
            cost = fill_price * order.qty
            self._balance[quote] = self._balance.get(quote, 0.0) - cost
            self._balance[base] = self._balance.get(base, 0.0) + order.qty
        else:
            proceeds = fill_price * order.qty
            self._balance[quote] = self._balance.get(quote, 0.0) + proceeds
            self._balance[base] = self._balance.get(base, 0.0) - order.qty

        logger.info(
            "[PAPER] Filled %s %s %.6f @ %.4f | balance=%s",
            order.side.upper(),
            order.symbol,
            order.qty,
            fill_price,
            {k: f"{v:.4f}" for k, v in self._balance.items()},
        )
        return order

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        if order_id in self._open_orders:
            self._open_orders[order_id].status = "cancelled"
            del self._open_orders[order_id]
            return True
        return False

    def get_balance(self) -> dict[str, float]:
        return dict(self._balance)

    def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        orders = list(self._open_orders.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders


def symbol_parts(symbol: str) -> tuple[str, str]:
    """Split 'BTC/USDT' → ('USDT', 'BTC')."""
    parts = symbol.split("/")
    if len(parts) == 2:
        return parts[1], parts[0]
    return "USDT", symbol
