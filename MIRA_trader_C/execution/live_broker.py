"""
MIRA_trader_C – Live Broker (CCXT).

Places real orders on the exchange.
Requires valid API credentials set via environment variables.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from execution.base import BrokerBase, Order, OrderSide

logger = logging.getLogger(__name__)


class LiveBroker(BrokerBase):
    """Execute orders via CCXT on a real exchange."""

    def __init__(self, exchange_cfg: dict[str, Any]) -> None:
        try:
            import ccxt  # type: ignore
        except ImportError as exc:
            raise ImportError("ccxt is required: pip install ccxt") from exc

        exchange_id = exchange_cfg.get("name", "binance")
        exchange_class = getattr(ccxt, exchange_id)
        self._exchange = exchange_class(
            {
                "apiKey": exchange_cfg.get("api_key", ""),
                "secret": exchange_cfg.get("api_secret", ""),
                "enableRateLimit": True,
            }
        )
        if exchange_cfg.get("testnet", False):
            self._exchange.set_sandbox_mode(True)

        logger.info("LiveBroker initialised on %s (testnet=%s)", exchange_id, exchange_cfg.get("testnet"))

    def place_market_order(self, symbol: str, side: OrderSide, qty: float) -> Order:
        logger.info("[LIVE] Placing market %s %s qty=%.6f", side, symbol, qty)
        raw = self._exchange.create_market_order(symbol, side, qty)
        order = Order(
            order_id=str(raw.get("id", uuid.uuid4())),
            symbol=symbol,
            side=side,
            qty=qty,
            price=raw.get("price") or raw.get("average") or 0.0,
            status="filled" if raw.get("status") == "closed" else "open",
            filled_qty=raw.get("filled", 0.0),
            filled_price=raw.get("average") or raw.get("price") or 0.0,
        )
        logger.info("[LIVE] Order result: %s", raw)
        return order

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        try:
            self._exchange.cancel_order(order_id, symbol)
            return True
        except Exception as exc:
            logger.warning("cancel_order failed: %s", exc)
            return False

    def get_balance(self) -> dict[str, float]:
        raw = self._exchange.fetch_balance()
        return {asset: info["free"] for asset, info in raw["total"].items() if info > 0}

    def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        raw_orders = self._exchange.fetch_open_orders(symbol)
        orders = []
        for raw in raw_orders:
            orders.append(
                Order(
                    order_id=str(raw["id"]),
                    symbol=raw["symbol"],
                    side=raw["side"],
                    qty=raw["amount"],
                    price=raw["price"] or 0.0,
                    status="open",
                )
            )
        return orders
