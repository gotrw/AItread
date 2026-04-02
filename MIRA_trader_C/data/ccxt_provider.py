"""
MIRA_trader_C – CCXT-based data provider (free public market data).

Uses the open-source `ccxt` library to fetch OHLCV candles from any supported
exchange using only public (unauthenticated) endpoints for price data.
No paid subscription is required.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from .base import DataProvider

logger = logging.getLogger(__name__)


class CCXTDataProvider(DataProvider):
    """Fetch market data via CCXT (supports 100+ exchanges)."""

    def __init__(self, exchange_cfg: dict[str, Any]) -> None:
        try:
            import ccxt  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "ccxt is required for CCXTDataProvider. "
                "Install it with: pip install ccxt"
            ) from exc

        exchange_id: str = exchange_cfg.get("name", "binance")
        exchange_class = getattr(ccxt, exchange_id)

        init_params: dict[str, Any] = {}
        api_key = exchange_cfg.get("api_key", "")
        api_secret = exchange_cfg.get("api_secret", "")
        if api_key and api_secret:
            init_params["apiKey"] = api_key
            init_params["secret"] = api_secret

        if exchange_cfg.get("testnet", False):
            init_params["options"] = {"defaultType": "future"}

        self._exchange = exchange_class(init_params)
        logger.info("CCXTDataProvider initialised with exchange: %s", exchange_id)

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
    ) -> pd.DataFrame:
        """Fetch OHLCV candles and return as a tidy DataFrame."""
        raw = self._exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(
            raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df = df.astype(float)
        logger.debug("Fetched %d candles for %s [%s]", len(df), symbol, timeframe)
        return df

    def fetch_ticker(self, symbol: str) -> dict:
        ticker = self._exchange.fetch_ticker(symbol)
        return {
            "last": ticker.get("last"),
            "bid": ticker.get("bid"),
            "ask": ticker.get("ask"),
            "volume": ticker.get("baseVolume"),
        }
