"""
MIRA_trader_C – Configuration Loader
Reads config/config.yaml and merges environment-variable overrides for secrets.
Supports per-symbol overrides from config/symbols/<SYMBOL>.yaml.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"
_SYMBOLS_DIR = Path(__file__).parent / "symbols"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and return the configuration dictionary.

    API credentials are injected from environment variables
    (MIRA_API_KEY / MIRA_API_SECRET) so secrets are never stored in files.
    Telegram credentials are injected from MIRA_TELEGRAM_TOKEN / MIRA_TELEGRAM_CHAT_ID.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)

    # Override exchange credentials from environment variables
    api_key = os.environ.get("MIRA_API_KEY", "")
    api_secret = os.environ.get("MIRA_API_SECRET", "")
    if api_key:
        cfg.setdefault("exchange", {})["api_key"] = api_key
    if api_secret:
        cfg.setdefault("exchange", {})["api_secret"] = api_secret

    # Override Telegram credentials from environment variables
    tg_token = os.environ.get("MIRA_TELEGRAM_TOKEN", "")
    tg_chat_id = os.environ.get("MIRA_TELEGRAM_CHAT_ID", "")
    if tg_token:
        cfg.setdefault("notifications", {}).setdefault("telegram", {})["bot_token"] = tg_token
    if tg_chat_id:
        cfg.setdefault("notifications", {}).setdefault("telegram", {})["chat_id"] = tg_chat_id

    return cfg


def load_symbol_config(symbol: str, global_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load per-symbol config merged on top of the global config.

    Symbol name is normalised (e.g. 'BTC/USDT' → 'BTC_USDT') to match filenames.
    Keys present in the symbol file override the global config; missing keys
    inherit global defaults.

    Args:
        symbol: Trading pair, e.g. 'BTC/USDT' or 'BTCUSDT'.
        global_cfg: Already-loaded global config dict. If None, loads from default path.

    Returns:
        Merged config dict for the symbol.
    """
    if global_cfg is None:
        global_cfg = load_config()

    # Deep-copy so we don't mutate the shared global config
    merged = copy.deepcopy(global_cfg)

    # Normalise symbol name to filename (BTC/USDT → BTC_USDT)
    filename_stem = symbol.replace("/", "_").replace(":", "_")
    symbol_path = _SYMBOLS_DIR / f"{filename_stem}.yaml"

    if not symbol_path.exists():
        return merged

    with open(symbol_path, "r", encoding="utf-8") as fh:
        overrides: dict[str, Any] = yaml.safe_load(fh) or {}

    _deep_merge(merged, overrides)
    return merged


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge *override* into *base* in-place.

    Nested dicts are merged recursively; scalar values are replaced.
    """
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val

