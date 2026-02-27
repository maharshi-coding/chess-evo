"""Tests for utility helpers (config loader, logger)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.utils import config as cfg_module


class TestLoadConfig:
    def test_returns_dict(self, tmp_path):
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("engine:\n  skill_level: 7\n")
        # Reset lru_cache between tests.
        cfg_module.load_config.cache_clear()
        data = cfg_module.load_config(str(cfg_file))
        assert isinstance(data, dict)
        assert data["engine"]["skill_level"] == 7

    def test_missing_file_returns_empty(self, tmp_path):
        cfg_module.load_config.cache_clear()
        data = cfg_module.load_config(str(tmp_path / "nonexistent.yaml"))
        assert data == {}

    def test_result_is_cached(self, tmp_path):
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("x: 1\n")
        cfg_module.load_config.cache_clear()
        d1 = cfg_module.load_config(str(cfg_file))
        d2 = cfg_module.load_config(str(cfg_file))
        assert d1 is d2  # same object – cache hit


class TestGet:
    def _setup(self, tmp_path, data: dict):
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text(yaml.dump(data))
        cfg_module.load_config.cache_clear()
        # Monkey-patch load_config to use our file.
        import functools
        original = cfg_module.load_config
        cfg_module.load_config = functools.lru_cache(maxsize=1)(
            lambda path=None: yaml.safe_load(cfg_file.read_text()) or {}
        )
        return original

    def teardown_method(self):
        cfg_module.load_config.cache_clear()

    def test_simple_key(self, tmp_path):
        orig = self._setup(tmp_path, {"foo": 42})
        try:
            assert cfg_module.get("foo") == 42
        finally:
            cfg_module.load_config = orig

    def test_nested_key(self, tmp_path):
        orig = self._setup(tmp_path, {"a": {"b": {"c": 99}}})
        try:
            assert cfg_module.get("a.b.c") == 99
        finally:
            cfg_module.load_config = orig

    def test_missing_key_returns_default(self, tmp_path):
        orig = self._setup(tmp_path, {})
        try:
            assert cfg_module.get("does.not.exist", "fallback") == "fallback"
        finally:
            cfg_module.load_config = orig
