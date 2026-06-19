"""Configuration for the planning service.

Two layers, highest priority first:

1. ``PLANNING_LLM_*`` environment variables (override everything; useful
   for secrets and per-process overrides).
2. ``planning:`` block in the YAML config (``configs/dev.yaml`` by
   default). Holds the per-deployment defaults — base URL, model name,
   timeout, temperature.

``api_key`` deliberately lives in env only — it must never end up in
version control. The YAML is allowed to reference an env var name via
``api_key_env: PLANNING_LLM_API_KEY`` if a deployment wants the
non-secret bits in YAML but the secret resolved from env.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class PlannerConfig:
    base_url: str
    model: str
    api_key: str | None
    timeout_s: float
    temperature: float


DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_TEMPERATURE = 0.0


def load_config(
    yaml_path: str | Path | None = None,
    yaml_data: Mapping[str, Any] | None = None,
) -> PlannerConfig:
    """Build a ``PlannerConfig``.

    Resolution order (first non-empty wins, per field):
      1. ``PLANNING_LLM_*`` env var
      2. ``planning:`` block in ``yaml_path`` (defaults to
         ``configs/dev.yaml``)
      3. Built-in default

    If ``yaml_data`` is supplied (e.g. in tests), it is used instead of
    reading from disk.
    """
    planning = _load_planning_block(yaml_path, yaml_data)

    base_url = (
        os.getenv("PLANNING_LLM_BASE_URL")
        or planning.get("base_url")
        or DEFAULT_BASE_URL
    ).rstrip("/")
    model = (
        os.getenv("PLANNING_LLM_MODEL")
        or planning.get("model")
        or DEFAULT_MODEL
    )
    api_key = _resolve_api_key(planning)
    timeout_s = float(
        os.getenv("PLANNING_LLM_TIMEOUT_S")
        or planning.get("timeout_s")
        or DEFAULT_TIMEOUT_S
    )
    temperature = float(
        os.getenv("PLANNING_LLM_TEMPERATURE")
        if os.getenv("PLANNING_LLM_TEMPERATURE") is not None
        else planning.get("temperature", DEFAULT_TEMPERATURE)
    )
    return PlannerConfig(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout_s=timeout_s,
        temperature=temperature,
    )


def _load_planning_block(
    yaml_path: str | Path | None,
    yaml_data: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if yaml_data is not None:
        return dict(yaml_data.get("planning", {}) or {})
    if yaml_path is None:
        yaml_path = _default_yaml_path()
    try:
        import yaml
    except ImportError:
        return {}
    path = Path(yaml_path)
    if not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text()) or {}
    return dict(loaded.get("planning", {}) or {})


def _default_yaml_path() -> Path:
    # configs/dev.yaml is the project's canonical config; planning lives
    # there under the ``planning:`` key.
    return Path(__file__).resolve().parents[3] / "configs" / "dev.yaml"


def _resolve_api_key(planning: Mapping[str, Any]) -> str | None:
    """Resolve the API key from env, with optional ``api_key_env`` indirection.

    The YAML block may set ``api_key_env: SOME_ENV_VAR`` to declare which
    env var holds the secret; the actual value is always read from the
    environment so secrets never live in the YAML file.
    """
    direct = os.getenv("PLANNING_LLM_API_KEY")
    if direct:
        return direct
    ref = planning.get("api_key_env")
    if ref:
        return os.getenv(ref) or None
    return None
