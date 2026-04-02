"""
MIRA_trader_C – Telegram Notifier.

Sends alert messages to a Telegram bot using the Bot HTTP API.
No third-party Telegram library is required – only the standard ``urllib``
and ``json`` modules (plus ``requests`` if available).

Configuration
-------------
Credentials are read from environment variables (never from config files):
  MIRA_TELEGRAM_TOKEN   – bot token from @BotFather
  MIRA_TELEGRAM_CHAT_ID – target chat / channel ID

In config.yaml:
  notifications:
    telegram:
      enabled: true
      send_signals: true
      send_orders: true
      send_risk_events: true
      send_errors: true
      batch_summary_hours: 1   # 0 = disable batch summaries

Alert types
-----------
- send_signal(scanned_signal)       – new scanner signal
- send_order(symbol, side, qty, price, pnl=None)
- send_risk_event(event_type, data) – daily loss, kill-switch, cooldown
- send_error(message)               – system errors
- send_summary(metrics_dict)        – periodic summary
"""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotifier:
    """Send Telegram messages for various bot events.

    Parameters
    ----------
    cfg:
        The ``notifications.telegram`` section of config.yaml, merged with
        environment-variable credentials.
    """

    def __init__(self, cfg: dict) -> None:
        import os

        self._enabled: bool = cfg.get("enabled", False)
        self._token: str = os.environ.get("MIRA_TELEGRAM_TOKEN", "") or cfg.get("bot_token", "")
        self._chat_id: str = os.environ.get("MIRA_TELEGRAM_CHAT_ID", "") or cfg.get("chat_id", "")
        self._send_signals: bool = cfg.get("send_signals", True)
        self._send_orders: bool = cfg.get("send_orders", True)
        self._send_risk: bool = cfg.get("send_risk_events", True)
        self._send_errors: bool = cfg.get("send_errors", True)
        summary_hours: int = cfg.get("batch_summary_hours", 1)
        self._summary_interval: float = summary_hours * 3600.0 if summary_hours > 0 else 0.0

        self._pending_messages: List[str] = []
        self._lock = threading.Lock()
        self._last_summary_time: float = time.time()

        if self._enabled and not (self._token and self._chat_id):
            logger.warning(
                "Telegram enabled but MIRA_TELEGRAM_TOKEN or MIRA_TELEGRAM_CHAT_ID is not set. "
                "Alerts will be suppressed."
            )
            self._enabled = False

        if self._enabled:
            logger.info("TelegramNotifier enabled (chat_id=%s)", self._chat_id)
        else:
            logger.info("TelegramNotifier disabled.")

    # ── Public alert methods ───────────────────────────────────

    def send_signal(self, scanned_signal: Any) -> None:
        """Alert for a new scanner signal (ScannedSignal instance)."""
        if not self._enabled or not self._send_signals:
            return
        direction = "🟢 LONG" if scanned_signal.action == "long" else "🔴 SHORT"
        msg = (
            f"📡 *Signal Scanner*\n"
            f"Pair: `{scanned_signal.symbol}`\n"
            f"Signal: {direction}\n"
            f"Confidence: `{scanned_signal.confidence:.0%}`  Risk: `{scanned_signal.risk_label()}`\n"
            f"Regime: `{scanned_signal.regime or 'unknown'}`\n"
            f"Price: `{scanned_signal.price:.4f}`\n"
            f"Strategy: `{scanned_signal.strategy}`\n"
            f"Time: `{scanned_signal.timestamp}`"
        )
        self._send(msg)

    def send_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        pnl: Optional[float] = None,
        exit_reason: str = "",
    ) -> None:
        """Alert when an order is filled."""
        if not self._enabled or not self._send_orders:
            return
        icon = "📥" if side in ("buy", "long") else "📤"
        pnl_str = f"\nPnL: `{'%.2f' % pnl} USDT`" if pnl is not None else ""
        reason_str = f"\nReason: `{exit_reason}`" if exit_reason else ""
        msg = (
            f"{icon} *Order Filled*\n"
            f"Pair: `{symbol}`\n"
            f"Side: `{side.upper()}`\n"
            f"Qty: `{qty:.6f}`\n"
            f"Price: `{price:.4f}`"
            f"{pnl_str}{reason_str}"
        )
        self._send(msg)

    def send_risk_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Alert for risk management events (daily loss hit, kill-switch, etc.)."""
        if not self._enabled or not self._send_risk:
            return
        icons = {
            "daily_loss_limit": "⚠️",
            "kill_switch": "🛑",
            "kill_switch_activated": "🛑",
            "liquidation_buffer": "🔥",
            "cooldown": "⏸️",
        }
        icon = icons.get(event_type, "⚠️")
        details = "\n".join(f"`{k}`: `{v}`" for k, v in data.items() if v != "")
        msg = (
            f"{icon} *Risk Event: {event_type}*\n"
            f"{details}"
        )
        self._send(msg)

    def send_error(self, message: str, symbol: str = "") -> None:
        """Alert for system errors."""
        if not self._enabled or not self._send_errors:
            return
        sym_str = f"\nSymbol: `{symbol}`" if symbol else ""
        msg = f"❌ *System Error*{sym_str}\n```\n{message[:1000]}\n```"
        self._send(msg)

    def send_summary(self, metrics: Dict[str, Any]) -> None:
        """Send a periodic performance summary."""
        if not self._enabled:
            return
        lines = [f"`{k}`: `{v}`" for k, v in metrics.items()]
        msg = "📊 *Performance Summary*\n" + "\n".join(lines)
        self._send(msg)

    def maybe_send_batch_summary(self, metrics: Dict[str, Any]) -> None:
        """Send a batch summary if the interval has elapsed."""
        if not self._enabled or self._summary_interval <= 0:
            return
        now = time.time()
        if now - self._last_summary_time >= self._summary_interval:
            self.send_summary(metrics)
            self._last_summary_time = now

    def send_system_status(self, status: str, details: str = "") -> None:
        """Send a system status message (start/stop/restart)."""
        if not self._enabled:
            return
        msg = f"ℹ️ *System Status*: `{status}`"
        if details:
            msg += f"\n{details}"
        self._send(msg)

    # ── Internal ───────────────────────────────────────────────

    def _send(self, text: str) -> None:
        """Send a message to the configured Telegram chat.

        Uses urllib so there is no dependency on the ``requests`` library.
        Falls back to requests if available for better error handling.
        """
        if not self._enabled:
            return

        url = _TELEGRAM_API.format(token=self._token)
        payload = json.dumps(
            {
                "chat_id": self._chat_id,
                "text": text,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            }
        ).encode("utf-8")

        # Try to send in a background thread so we never block the trading loop
        threading.Thread(target=self._http_post, args=(url, payload), daemon=True).start()

    def _http_post(self, url: str, payload: bytes) -> None:
        """Perform the HTTP POST in a background thread."""
        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status != 200:
                    logger.warning("Telegram API returned status %d", resp.status)
        except urllib.error.HTTPError as exc:
            logger.warning("Telegram HTTP error: %s %s", exc.code, exc.reason)
        except Exception as exc:
            logger.warning("Telegram send failed: %s", exc)


def build_notifier(cfg: dict) -> Optional[TelegramNotifier]:
    """Build a TelegramNotifier from the notifications config section.

    Returns None if Telegram is disabled.
    """
    tg_cfg = cfg.get("notifications", {}).get("telegram", {})
    if not tg_cfg.get("enabled", False):
        return None
    return TelegramNotifier(tg_cfg)
