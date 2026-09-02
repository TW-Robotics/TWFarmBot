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
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

try:
    from dotenv import load_dotenv

    # Load .env once at import time, but never override existing env vars
    # so that explicitly exported variables still win. Skip when running
    # under pytest (including collection-time imports, where
    # PYTEST_CURRENT_TEST is not yet set) to avoid triggering external
    # services (e.g. Weave) or leaking developer .env values into tests.
    if "pytest" not in sys.modules:
        load_dotenv(override=False)
except ImportError:  # pragma: no cover - python-dotenv is optional
    pass

from twfarmbot_core.config import load_yaml_config


@dataclass(frozen=True)
class PlannerConfig:
    provider: str
    base_url: str
    model: str
    api_key: str | None
    timeout_s: float
    temperature: float
    extra_body: dict[str, Any] | None = None
    weave_project: str | None = None
    permissive_provider: bool = False


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-5.6"
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

    provider = (
        os.getenv("PLANNING_LLM_PROVIDER") or planning.get("provider") or "openai"
    ).lower()
    base_url = (
        os.getenv("PLANNING_LLM_BASE_URL")
        or planning.get("base_url")
        or DEFAULT_BASE_URL
    ).rstrip("/")
    model = os.getenv("PLANNING_LLM_MODEL") or planning.get("model") or DEFAULT_MODEL
    api_key = _resolve_api_key(planning)
    timeout_s = float(
        os.getenv("PLANNING_LLM_TIMEOUT_S")
        or planning.get("timeout_s")
        or DEFAULT_TIMEOUT_S
    )
    raw_temperature = (
        os.getenv("PLANNING_LLM_TEMPERATURE")
        if os.getenv("PLANNING_LLM_TEMPERATURE") is not None
        else planning.get("temperature") or DEFAULT_TEMPERATURE
    )
    temperature = float(str(raw_temperature))
    extra_body = planning.get("extra_body")
    if extra_body is not None and not isinstance(extra_body, dict):
        extra_body = None
    weave_project = os.getenv("WEAVE_PROJECT") or planning.get("weave_project")
    return PlannerConfig(
        provider=provider,
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout_s=timeout_s,
        temperature=temperature,
        extra_body=extra_body,
        weave_project=weave_project,
    )


def _load_planning_block(
    yaml_path: str | Path | None,
    yaml_data: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if yaml_data is not None:
        return dict(yaml_data.get("planning", {}) or {})
    return dict(load_yaml_config(yaml_path).get("planning", {}) or {})


def apply_overrides(cfg: PlannerConfig, **overrides: Any) -> PlannerConfig:
    """Return a copy of ``cfg`` with non-``None`` override fields applied."""
    updates: dict[str, Any] = {}
    for key, value in overrides.items():
        if value is None:
            continue
        if key == "base_url":
            updates[key] = str(value).rstrip("/")
        elif key in {"timeout_s", "temperature"}:
            updates[key] = float(value)
        elif key == "provider":
            updates[key] = str(value).lower()
            updates["permissive_provider"] = True
        else:
            updates[key] = value
    return replace(cfg, **updates) if updates else cfg


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
