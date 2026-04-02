"""
MIRA_trader_C – Live Broker (CCXT).

Places real orders on the exchange.
Requires valid API credentials set via environment variables.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Callable, TypeVar

from execution.base import BrokerBase, Order, OrderSide

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

_MAX_RETRIES = 3
_RETRY_DELAY_SECONDS = 2.0


def _with_retry(fn: Callable[[], _T], *, retries: int = _MAX_RETRIES) -> _T:
    """Call *fn* and retry up to *retries* times on transient network errors."""
    try:
        import ccxt  # type: ignore

        retryable: tuple[type[Exception], ...] = (ccxt.NetworkError, ccxt.RequestTimeout)
    except ImportError:
        retryable = ()

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return fn()
        except retryable as exc:  # type: ignore[misc]
            last_exc = exc
            wait = _RETRY_DELAY_SECONDS * (attempt + 1)
            logger.warning(
                "Transient exchange error (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1,
                retries,
                exc,
                wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"Exchange call failed after {retries} retries") from last_exc


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
        raw = _with_retry(lambda: self._exchange.create_market_order(symbol, side, qty))
        filled_qty = float(raw.get("filled") or 0.0)
        filled_price = float(raw.get("average") or raw.get("price") or 0.0)
        order = Order(
            order_id=str(raw.get("id", uuid.uuid4())),
            symbol=symbol,
            side=side,
            qty=qty,
            price=filled_price,
            status="filled" if raw.get("status") == "closed" else "open",
            filled_qty=filled_qty,
            filled_price=filled_price,
        )
        logger.info("[LIVE] Order result: %s", raw)
        return order

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        try:
            _with_retry(lambda: self._exchange.cancel_order(order_id, symbol))
            return True
        except Exception as exc:
            logger.warning("cancel_order failed: %s", exc)
            return False

    def get_balance(self) -> dict[str, float]:
        """Return free (available) balances for all assets with a non-zero amount."""
        raw = _with_retry(lambda: self._exchange.fetch_balance())
        free: dict[str, Any] = raw.get("free") or {}
        return {
            asset: float(amount)
            for asset, amount in free.items()
            if amount is not None and float(amount) > 0
        }

    def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        raw_orders = _with_retry(lambda: self._exchange.fetch_open_orders(symbol))
        orders = []
        for raw in raw_orders:
            orders.append(
                Order(
                    order_id=str(raw["id"]),
                    symbol=raw["symbol"],
                    side=raw["side"],
                    qty=float(raw["amount"] or 0.0),
                    price=float(raw["price"] or 0.0),
                    status="open",
                )
            )
        return orders
