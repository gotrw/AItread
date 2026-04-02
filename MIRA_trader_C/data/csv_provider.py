"""
MIRA_trader_C – CSV-based data provider (offline / backtesting).

Loads pre-downloaded OHLCV data from CSV files.
Expected CSV format: timestamp,open,high,low,close,volume
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .base import DataProvider

logger = logging.getLogger(__name__)


class CSVDataProvider(DataProvider):
    """Load historical OHLCV data from local CSV files."""

    def __init__(self, csv_dir: str = "data/csv") -> None:
        self._csv_dir = Path(csv_dir)

    def _symbol_to_filename(self, symbol: str, timeframe: str) -> Path:
        safe_symbol = symbol.replace("/", "_")
        return self._csv_dir / f"{safe_symbol}_{timeframe}.csv"

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
    ) -> pd.DataFrame:
        fpath = self._symbol_to_filename(symbol, timeframe)
        if not fpath.exists():
            raise FileNotFoundError(
                f"CSV file not found: {fpath}. "
                "Download data first or switch to provider=ccxt."
            )
        df = pd.read_csv(fpath, parse_dates=["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.set_index("timestamp", inplace=True)
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        if limit and len(df) > limit:
            df = df.iloc[-limit:]
        logger.debug("Loaded %d rows from %s", len(df), fpath)
        return df

    def fetch_ticker(self, symbol: str) -> dict:
        raise NotImplementedError("CSVDataProvider does not support live tickers.")
