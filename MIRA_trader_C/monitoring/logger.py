"""
MIRA_trader_C – Structured Logging Setup.

Call `setup_logging(cfg)` once at startup to configure log level, console
handler, and rotating file handler with structured JSON formatting.
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class _JSONFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging(cfg: dict) -> None:
    """Configure root logger from monitoring config section."""
    mon_cfg = cfg.get("monitoring", {})
    level_name: str = mon_cfg.get("log_level", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # Remove existing handlers so this function is idempotent
    for handler in list(root.handlers):
        root.removeHandler(handler)

    # Console handler (human-readable)
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    )
    root.addHandler(console)

    # File handler (JSON, rotating)
    log_file: str = mon_cfg.get("log_file", "logs/mira.log")
    log_path = Path(log_file)
    os.makedirs(log_path.parent, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(_JSONFormatter())
    root.addHandler(file_handler)

    logging.getLogger(__name__).info("Logging initialised: level=%s file=%s", level_name, log_file)
