"""
MIRA_trader_C – Order Execution Abstraction.

Defines the common interface for executing orders so that strategies and
risk logic are fully decoupled from the execution back-end.

Concrete implementations:
  - PaperBroker  (no real orders, perfect fill at mid-price)
  - LiveBroker   (CCXT, real exchange orders)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

OrderStatus = Literal["open", "filled", "cancelled", "rejected"]
OrderSide = Literal["buy", "sell"]


@dataclass
class Order:
    """Represents a submitted order."""
    order_id: str
    symbol: str
    side: OrderSide
    qty: float
    price: float
    status: OrderStatus = "open"
    filled_qty: float = 0.0
    filled_price: float = 0.0
    meta: dict = field(default_factory=dict)


class BrokerBase(ABC):
    """Abstract broker interface."""

    @abstractmethod
    def place_market_order(self, symbol: str, side: OrderSide, qty: float) -> Order:
        """Submit a market order and return the filled Order object."""

    @abstractmethod
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order. Returns True if successful."""

    @abstractmethod
    def get_balance(self) -> dict[str, float]:
        """Return available balances, e.g. {'USDT': 1000.0, 'BTC': 0.001}."""

    @abstractmethod
    def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        """Return currently open orders."""
