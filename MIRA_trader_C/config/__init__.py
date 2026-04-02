"""
MIRA_trader_C – Configuration Loader
Reads config/config.yaml and merges environment-variable overrides for secrets.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and return the configuration dictionary.

    API credentials are injected from environment variables
    (MIRA_API_KEY / MIRA_API_SECRET) so secrets are never stored in files.
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

    return cfg
