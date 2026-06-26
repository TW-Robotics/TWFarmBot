"""Tests for ``planning_service.config.load_config``.

The config has three layers (env > YAML > default) and a separate code
path for the API key (always env, with optional ``api_key_env``
indirection from YAML).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from planning_service.config import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_S,
    load_config,
)


def test_defaults_when_nothing_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for var in (
        "PLANNING_LLM_BASE_URL",
        "PLANNING_LLM_MODEL",
        "PLANNING_LLM_API_KEY",
        "PLANNING_LLM_TIMEOUT_S",
        "PLANNING_LLM_TEMPERATURE",
    ):
        monkeypatch.delenv(var, raising=False)
    cfg = load_config(yaml_path=tmp_path / "missing.yaml")
    assert cfg.base_url == DEFAULT_BASE_URL
    assert cfg.model == DEFAULT_MODEL
    assert cfg.api_key is None
    assert cfg.timeout_s == DEFAULT_TIMEOUT_S
    assert cfg.temperature == DEFAULT_TEMPERATURE


def test_yaml_block_used_when_env_unset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("PLANNING_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("PLANNING_LLM_MODEL", raising=False)
    monkeypatch.delenv("PLANNING_LLM_API_KEY", raising=False)
    yaml_file = tmp_path / "cfg.yaml"
    yaml_file.write_text(
        "planning:\n"
        "  base_url: http://localhost:8000/v1\n"
        "  model: my-checkpoint\n"
        "  timeout_s: 10\n"
        "  temperature: 0.3\n"
    )
    cfg = load_config(yaml_path=yaml_file)
    assert cfg.base_url == "http://localhost:8000/v1"
    assert cfg.model == "my-checkpoint"
    assert cfg.timeout_s == 10.0
    assert cfg.temperature == 0.3


def test_env_overrides_yaml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PLANNING_LLM_BASE_URL", "https://override.example/v1")
    monkeypatch.setenv("PLANNING_LLM_MODEL", "override-model")
    monkeypatch.setenv("PLANNING_LLM_TIMEOUT_S", "5")
    yaml_file = tmp_path / "cfg.yaml"
    yaml_file.write_text(
        "planning:\n"
        "  base_url: http://localhost:8000/v1\n"
        "  model: my-checkpoint\n"
        "  timeout_s: 10\n"
    )
    cfg = load_config(yaml_path=yaml_file)
    assert cfg.base_url == "https://override.example/v1"
    assert cfg.model == "override-model"
    assert cfg.timeout_s == 5.0


def test_api_key_resolved_from_indirected_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("PLANNING_LLM_API_KEY", raising=False)
    monkeypatch.setenv("MY_PLANNER_KEY", "secret-from-env")
    yaml_file = tmp_path / "cfg.yaml"
    yaml_file.write_text("planning:\n  api_key_env: MY_PLANNER_KEY\n")
    cfg = load_config(yaml_path=yaml_file)
    assert cfg.api_key == "secret-from-env"


def test_api_key_direct_env_wins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PLANNING_LLM_API_KEY", "direct-key")
    monkeypatch.setenv("MY_PLANNER_KEY", "indirected-key")
    yaml_file = tmp_path / "cfg.yaml"
    yaml_file.write_text("planning:\n  api_key_env: MY_PLANNER_KEY\n")
    cfg = load_config(yaml_path=yaml_file)
    assert cfg.api_key == "direct-key"


def test_yaml_data_argument_skips_filesystem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PLANNING_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("PLANNING_LLM_MODEL", raising=False)
    cfg = load_config(
        yaml_data={
            "planning": {
                "base_url": "http://from-dict:1234/v1",
                "model": "dict-model",
            }
        }
    )
    assert cfg.base_url == "http://from-dict:1234/v1"
    assert cfg.model == "dict-model"


def test_real_dev_yaml_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shipped ``configs/dev.yaml`` must be a valid planner config."""
    monkeypatch.delenv("PLANNING_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("PLANNING_LLM_MODEL", raising=False)
    monkeypatch.delenv("PLANNING_LLM_API_KEY", raising=False)
    cfg = load_config()  # uses default path
    assert cfg.base_url.startswith("http")
    assert cfg.model  # non-empty
    assert cfg.timeout_s > 0
