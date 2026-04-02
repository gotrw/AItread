"""
MIRA_trader_C – Base DataProvider interface.

All concrete data providers (CCXT, CSV, …) implement this interface so that
strategies and the backtesting engine remain decoupled from the data source.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class DataProvider(ABC):
    """Abstract base class for market-data providers."""

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
    ) -> pd.DataFrame:
        """Return a DataFrame with columns: open, high, low, close, volume.

        The DataFrame index must be a UTC-aware DatetimeTzIndex.
        """

    @abstractmethod
    def fetch_ticker(self, symbol: str) -> dict:
        """Return a dict with at least {'last': float, 'bid': float, 'ask': float}."""
