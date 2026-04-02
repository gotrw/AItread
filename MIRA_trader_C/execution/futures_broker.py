"""
MIRA_trader_C – Futures Broker (CCXT).

Extends LiveBroker with futures-specific operations:
- Set leverage per symbol
- Set margin type (isolated / cross)
- Fetch liquidation price and margin info
- Place long/short market orders with proper side mapping
- Monitor funding rate

Requires a futures-enabled CCXT exchange (e.g. binanceusdm).
API credentials must be supplied via environment variables:
  MIRA_API_KEY, MIRA_API_SECRET
"""
from __future__ import annotations

import logging
from typing import Any

from execution.base import BrokerBase, Order, OrderSide
from execution.live_broker import LiveBroker

logger = logging.getLogger(__name__)


class FuturesBroker(LiveBroker):
    """CCXT-based futures broker for USDT-margined perpetual contracts."""

    def __init__(self, exchange_cfg: dict[str, Any]) -> None:
        # Force futures market type
        import ccxt  # type: ignore

        exchange_id = exchange_cfg.get("name", "binance")
        # Use binanceusdm for USDT-margined futures if binance spot is specified
        futures_id_map = {"binance": "binanceusdm", "okx": "okx", "bybit": "bybit"}
        futures_exchange_id = futures_id_map.get(exchange_id, exchange_id)

        exchange_class = getattr(ccxt, futures_exchange_id)
        self._exchange = exchange_class(
            {
                "apiKey": exchange_cfg.get("api_key", ""),
                "secret": exchange_cfg.get("api_secret", ""),
                "enableRateLimit": True,
                "options": {"defaultType": "future"},
            }
        )
        if exchange_cfg.get("testnet", False):
            self._exchange.set_sandbox_mode(True)

        self._futures_cfg = exchange_cfg
        logger.info(
            "FuturesBroker initialised on %s (testnet=%s)",
            futures_exchange_id,
            exchange_cfg.get("testnet"),
        )

    # ── Futures-specific methods ───────────────────────────────

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol.  Returns True on success."""
        try:
            self._exchange.set_leverage(leverage, symbol)
            logger.info("[FUTURES] Set leverage=%d for %s", leverage, symbol)
            return True
        except Exception as exc:
            logger.warning("set_leverage failed for %s: %s", symbol, exc)
            return False

    def set_margin_type(self, symbol: str, margin_type: str = "isolated") -> bool:
        """Set margin type: 'isolated' or 'cross'.  Returns True on success."""
        try:
            self._exchange.set_margin_mode(margin_type.upper(), symbol)
            logger.info("[FUTURES] Set margin_type=%s for %s", margin_type, symbol)
            return True
        except Exception as exc:
            logger.warning("set_margin_type failed for %s: %s", symbol, exc)
            return False

    def get_position_risk(self, symbol: str) -> dict:
        """Fetch position risk info including liquidation price.

        Returns a dict with keys:
            liquidation_price, unrealised_pnl, margin, leverage,
            position_side, entry_price, contracts
        """
        try:
            positions = self._exchange.fetch_positions([symbol])
            for pos in positions:
                if pos.get("symbol") == symbol and float(pos.get("contracts", 0) or 0) != 0:
                    return {
                        "liquidation_price": float(pos.get("liquidationPrice") or 0),
                        "unrealised_pnl": float(pos.get("unrealizedPnl") or 0),
                        "margin": float(pos.get("initialMargin") or 0),
                        "leverage": float(pos.get("leverage") or 1),
                        "position_side": pos.get("side", ""),
                        "entry_price": float(pos.get("entryPrice") or 0),
                        "contracts": float(pos.get("contracts") or 0),
                    }
        except Exception as exc:
            logger.warning("get_position_risk failed for %s: %s", symbol, exc)
        return {}

    def get_funding_rate(self, symbol: str) -> float:
        """Fetch the current funding rate for a symbol (0.0001 = 0.01%)."""
        try:
            funding = self._exchange.fetch_funding_rate(symbol)
            rate = float(funding.get("fundingRate") or 0)
            logger.debug("[FUTURES] Funding rate for %s: %.6f", symbol, rate)
            return rate
        except Exception as exc:
            logger.warning("get_funding_rate failed for %s: %s", symbol, exc)
            return 0.0

    def place_long_order(self, symbol: str, qty: float) -> Order:
        """Open a long (buy) position."""
        return self.place_market_order(symbol, "buy", qty)

    def place_short_order(self, symbol: str, qty: float) -> Order:
        """Open a short (sell) position."""
        return self.place_market_order(symbol, "sell", qty)

    def close_long(self, symbol: str, qty: float) -> Order:
        """Close a long position by selling."""
        logger.info("[FUTURES] Closing long %s qty=%.6f", symbol, qty)
        raw = self._exchange.create_market_order(
            symbol, "sell", qty, params={"reduceOnly": True}
        )
        return Order(
            order_id=str(raw.get("id", "")),
            symbol=symbol,
            side="sell",
            qty=qty,
            price=raw.get("average") or raw.get("price") or 0.0,
            status="filled" if raw.get("status") == "closed" else "open",
            filled_qty=raw.get("filled", 0.0),
            filled_price=raw.get("average") or raw.get("price") or 0.0,
        )

    def close_short(self, symbol: str, qty: float) -> Order:
        """Close a short position by buying."""
        logger.info("[FUTURES] Closing short %s qty=%.6f", symbol, qty)
        raw = self._exchange.create_market_order(
            symbol, "buy", qty, params={"reduceOnly": True}
        )
        return Order(
            order_id=str(raw.get("id", "")),
            symbol=symbol,
            side="buy",
            qty=qty,
            price=raw.get("average") or raw.get("price") or 0.0,
            status="filled" if raw.get("status") == "closed" else "open",
            filled_qty=raw.get("filled", 0.0),
            filled_price=raw.get("average") or raw.get("price") or 0.0,
        )


class PaperFuturesBroker(BrokerBase):
    """Simulated futures broker for paper trading.

    Tracks simulated long/short positions with leverage and margin.
    Computes a synthetic liquidation price based on position leverage.
    """

    def __init__(
        self,
        initial_balance: dict[str, float] | None = None,
        futures_cfg: dict | None = None,
    ) -> None:
        self._balance: dict[str, float] = initial_balance or {"USDT": 1000.0}
        self._positions: dict[str, dict] = {}  # symbol → position info
        self._futures_cfg = futures_cfg or {}
        import uuid
        self._uuid = uuid

    def place_market_order(self, symbol: str, side: OrderSide, qty: float) -> Order:
        import uuid as _uuid
        order_id = str(_uuid.uuid4())[:8]
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            qty=qty,
            price=0.0,
            status="open",
        )
        logger.info("[PAPER-FUTURES] Market %s %s qty=%.6f → %s", side.upper(), symbol, qty, order_id)
        return order

    def fill_paper_order(self, order: Order, fill_price: float, leverage: int = 1) -> Order:
        """Fill a paper order and record simulated position."""
        order.filled_qty = order.qty
        order.filled_price = fill_price
        order.price = fill_price
        order.status = "filled"

        # Compute liquidation price (approximate for isolated margin)
        margin_rate = 1.0 / leverage
        if order.side == "buy":
            liq_price = fill_price * (1 - margin_rate + 0.005)  # 0.5% maintenance margin
        else:
            liq_price = fill_price * (1 + margin_rate - 0.005)

        self._positions[order.symbol] = {
            "side": order.side,
            "qty": order.qty,
            "entry_price": fill_price,
            "leverage": leverage,
            "liquidation_price": liq_price,
            "unrealised_pnl": 0.0,
        }
        logger.info(
            "[PAPER-FUTURES] Filled %s %s %.6f @ %.4f liq=%.4f",
            order.side.upper(), order.symbol, order.qty, fill_price, liq_price,
        )
        return order

    def update_pnl(self, symbol: str, current_price: float) -> float:
        """Update unrealised PnL for an open position and return it."""
        if symbol not in self._positions:
            return 0.0
        pos = self._positions[symbol]
        qty = pos["qty"]
        entry = pos["entry_price"]
        leverage = pos["leverage"]
        if pos["side"] == "buy":
            pnl = (current_price - entry) * qty * leverage
        else:
            pnl = (entry - current_price) * qty * leverage
        pos["unrealised_pnl"] = pnl
        return pnl

    def is_liquidated(self, symbol: str, current_price: float) -> bool:
        """Return True if the position's liquidation price has been breached."""
        if symbol not in self._positions:
            return False
        pos = self._positions[symbol]
        liq = pos["liquidation_price"]
        if pos["side"] == "buy" and current_price <= liq:
            return True
        if pos["side"] == "sell" and current_price >= liq:
            return True
        return False

    def close_position(self, symbol: str) -> dict | None:
        """Remove and return the closed position record."""
        return self._positions.pop(symbol, None)

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        return False

    def get_balance(self) -> dict[str, float]:
        return dict(self._balance)

    def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        return []

    def get_positions(self) -> dict[str, dict]:
        return dict(self._positions)
