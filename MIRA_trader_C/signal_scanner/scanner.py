"""
MIRA_trader_C – Signal Scanner.

Runs an independent scan loop that analyses every configured symbol
and emits ScannedSignal objects regardless of whether auto-trading is active.
This decouples "what the market is doing" from "whether to place a trade".

Outputs
-------
- Appends every signal to ``logs/signal_log.jsonl``
- Pushes actionable signals to the shared ``_signal_store`` (used by /api/signals)
- Optionally sends Telegram alerts for actionable signals

Usage (standalone scanner mode)
--------------------------------
    scanner = SignalScanner(cfg, data_provider, strategy)
    asyncio.run(scanner.run())

Usage (integrated – scanner feeds web dashboard while bot trades)
-----------------------------------------------------------------
    scanner = SignalScanner(cfg, data_provider, strategy)
    asyncio.ensure_future(scanner.run())
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from data.base import DataProvider
from strategies.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


@dataclass
class ScannedSignal:
    """A signal produced by the scanner for a specific symbol."""

    symbol: str
    action: str           # long | short | hold
    confidence: float     # 0.0 – 1.0
    strategy: str         # originating strategy name
    regime: str = ""      # market regime at scan time
    price: float = 0.0    # price at scan time
    timestamp: str = ""   # ISO 8601 UTC (auto-filled if empty)
    timeframe: str = "1h" # e.g. '1h'
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def is_actionable(self) -> bool:
        return self.action in ("long", "short")

    def risk_label(self) -> str:
        """Human-readable risk label based on confidence.

        Returns 'HIGH', 'MED', 'LOW', or '' for non-actionable signals.
        """
        if not self.is_actionable():
            return ""
        if self.confidence >= 0.8:
            return "HIGH"
        if self.confidence >= 0.5:
            return "MED"
        return "LOW"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["risk_label"] = self.risk_label()
        return d


# ── In-process signal store (shared with web dashboard) ───────

_signal_store: List[ScannedSignal] = []
_MAX_SIGNALS = 500


def get_recent_signals(limit: int = 50) -> List[dict]:
    """Return the most recent scanned signals as dicts (for /api/signals)."""
    return [s.to_dict() for s in _signal_store[-limit:]]


def _clear_signals() -> None:
    """Clear the in-process signal store (used in tests)."""
    _signal_store.clear()


def _store_signal(sig: ScannedSignal) -> None:
    _signal_store.append(sig)
    if len(_signal_store) > _MAX_SIGNALS:
        del _signal_store[: len(_signal_store) - _MAX_SIGNALS]


# ── Scanner ────────────────────────────────────────────────────

class SignalScanner:
    """Scans all configured symbols and emits ScannedSignal events.

    Parameters
    ----------
    cfg:
        Full global config dict (from ``load_config``).
    data_provider:
        Instantiated DataProvider for fetching OHLCV data.
    strategy:
        Instantiated strategy (typically MultiStrategy).
    notifier:
        Optional TelegramNotifier instance. If provided, actionable signals
        are sent as Telegram messages.
    """

    def __init__(
        self,
        cfg: dict,
        data_provider: DataProvider,
        strategy: BaseStrategy,
        notifier: Optional[Any] = None,
    ) -> None:
        self._cfg = cfg
        self._provider = data_provider
        self._strategy = strategy
        self._notifier = notifier

        scanner_cfg = cfg.get("scanner", {})
        self._interval: int = scanner_cfg.get("interval_seconds", 300)
        self._min_confidence: float = scanner_cfg.get("min_confidence", 0.3)
        self._send_telegram: bool = scanner_cfg.get("send_telegram", True)

        data_cfg = cfg.get("data", {})
        self._symbols: List[str] = data_cfg.get("symbols", ["BTC/USDT"])
        self._timeframe: str = data_cfg.get("timeframe", "1h")
        self._lookback: int = data_cfg.get("lookback_candles", 500)

        log_path = scanner_cfg.get("signal_log_file", "logs/signal_log.jsonl")
        self._log_path = Path(log_path)
        os.makedirs(self._log_path.parent, exist_ok=True)

    async def run(self) -> None:
        """Run the scanner loop indefinitely (until cancelled)."""
        logger.info(
            "SignalScanner started. Symbols=%s interval=%ds", self._symbols, self._interval
        )
        while True:
            await self._scan_all()
            await asyncio.sleep(self._interval)

    async def _scan_all(self) -> None:
        """Scan all symbols once and emit signals."""
        tasks = [
            asyncio.get_event_loop().run_in_executor(None, self._scan_symbol, sym)
            for sym in self._symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for sym, result in zip(self._symbols, results):
            if isinstance(result, Exception):
                logger.error("Scanner error for %s: %s", sym, result)

    def _scan_symbol(self, symbol: str) -> Optional[ScannedSignal]:
        """Scan a single symbol (runs in thread-pool executor)."""
        try:
            from config import load_symbol_config
            sym_cfg = load_symbol_config(symbol, self._cfg)
            timeframe = sym_cfg.get("data", {}).get("timeframe", self._timeframe)
            lookback = sym_cfg.get("data", {}).get("lookback_candles", self._lookback)

            df = self._provider.fetch_ohlcv(symbol, timeframe, lookback)
            if df.empty or len(df) < 30:
                logger.warning("Scanner: insufficient data for %s", symbol)
                return None

            signal: Signal = self._strategy.generate_signal(df)
            price = float(df["close"].iloc[-1])
            ts = datetime.now(tz=timezone.utc).isoformat()

            scanned = ScannedSignal(
                symbol=symbol,
                action=signal.action,
                confidence=signal.confidence,
                strategy=signal.strategy_name or self._strategy.name,
                regime=signal.regime,
                price=price,
                timestamp=ts,
                timeframe=timeframe,
                meta=signal.meta,
            )

            _store_signal(scanned)
            self._log_signal(scanned)

            if scanned.is_actionable() and scanned.confidence >= self._min_confidence:
                logger.info(
                    "SIGNAL [%s] %s @ %.4f conf=%.2f risk=%s regime=%s",
                    symbol, signal.action.upper(), price,
                    signal.confidence, scanned.risk_label(), signal.regime,
                )
                if self._send_telegram and self._notifier is not None:
                    self._notifier.send_signal(scanned)
            else:
                logger.debug("Scanner [%s] → %s (conf=%.2f)", symbol, signal.action, signal.confidence)

            return scanned

        except Exception as exc:
            logger.error("Scanner error for %s: %s", symbol, exc, exc_info=True)
            return None

    def scan_once(self) -> List[ScannedSignal]:
        """Perform a synchronous single scan of all symbols.

        Useful for testing or CLI usage where an event loop may not be running.
        """
        results = []
        for symbol in self._symbols:
            sig = self._scan_symbol(symbol)
            if sig is not None:
                results.append(sig)
        return results

    def _log_signal(self, sig: ScannedSignal) -> None:
        """Append a signal to the JSONL signal log file."""
        try:
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(sig.to_dict()) + "\n")
        except OSError as exc:
            logger.warning("Failed to write signal log: %s", exc)
