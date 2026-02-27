"""Configuration loader – reads config.yaml and exposes a single dict."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict[str, Any]:
    """Load and cache configuration from *path* (defaults to repo-root config.yaml).

    The result is cached after the first load so repeated calls are O(1).
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    if not config_path.exists():
        logger.warning("Config file %s not found – using empty config.", config_path)
        return {}
    with config_path.open("r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    logger.debug("Loaded config from %s", config_path)
    return data


def get(key_path: str, default: Any = None) -> Any:
    """Retrieve a nested config value using dot-notation key path.

    Example::

        get("engine.skill_level", 10)
    """
    cfg = load_config()
    keys = key_path.split(".")
    node: Any = cfg
    for k in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(k)
        if node is None:
            return default
    return node
