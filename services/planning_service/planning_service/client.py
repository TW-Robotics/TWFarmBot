"""LangChain chat-model factory for the planner.

We use ``langchain_openai.ChatOpenAI`` for both OpenRouter and any
self-hosted OpenAI-compatible endpoint (llama.cpp, vLLM, Ollama with
``OLLAMA_OPENAI_COMPAT=true``, TGI, etc.). The transport is identical
between them — the only knobs are ``base_url`` and ``model``.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel


def build_chat_model(
    *,
    base_url: str,
    model: str,
    api_key: str | None = None,
    timeout_s: float = 30.0,
    **overrides: Any,
) -> BaseChatModel:
    """Return a LangChain chat model pointed at an OpenAI-compatible API.

    The returned model is configured to return JSON. We do not pass
    ``response_format={"type": "json_object"}`` when the backend may not
    support it (some self-hosted servers ignore it); instead, the parser
    in :mod:`parser` is robust to free-form output.
    """
    # Local import keeps planning_service importable without langchain
    # installed (e.g. for type-checking or partial unit tests).
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {
        "base_url": base_url,
        "model": model,
        "timeout": timeout_s,
    }
    if api_key:
        kwargs["api_key"] = api_key
    kwargs.update(overrides)
    return ChatOpenAI(**kwargs)
