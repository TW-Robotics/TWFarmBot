"""LangChain Chat Completions factory for JSON function/tool calling.

We use ``langchain_openai.ChatOpenAI`` for OpenAI, OpenRouter, and any
self-hosted OpenAI-compatible endpoint (llama.cpp, vLLM, Ollama with
``OLLAMA_OPENAI_COMPAT=true``, TGI, etc.). The transport is identical
between them — the only knobs are ``base_url`` and ``model``.
"""

from __future__ import annotations

from typing import Any, Mapping

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
)


def _patch_langchain_openai_reasoning() -> None:
    """Capture provider-specific reasoning fields that LangChain drops.

    OpenRouter's DeepSeek V4 models return the chain-of-thought in a
    top-level ``reasoning`` field on the message. The standard LangChain
    OpenAI converter ignores it, so we patch the converters to copy it
    into ``additional_kwargs["reasoning"]`` where the rest of the code
    already looks.
    """
    from langchain_openai.chat_models.base import (
        _convert_delta_to_message_chunk,
        _convert_dict_to_message,
    )

    _orig_dict = _convert_dict_to_message
    _orig_delta = _convert_delta_to_message_chunk

    def _dict_with_reasoning(_dict: Mapping[str, Any]) -> BaseMessage:
        msg = _orig_dict(_dict)
        if isinstance(msg, AIMessage):
            reasoning = _dict.get("reasoning") or _dict.get("reasoning_content")
            if reasoning:
                msg.additional_kwargs.setdefault("reasoning", reasoning)
        return msg

    def _delta_with_reasoning(
        _dict: Mapping[str, Any], default_class: type[BaseMessageChunk]
    ) -> BaseMessageChunk:
        msg = _orig_delta(_dict, default_class)
        if isinstance(msg, AIMessageChunk):
            reasoning = _dict.get("reasoning") or _dict.get("reasoning_content")
            if reasoning:
                msg.additional_kwargs.setdefault("reasoning", reasoning)
        return msg

    import langchain_openai.chat_models.base as _base

    _base._convert_dict_to_message = _dict_with_reasoning
    _base._convert_delta_to_message_chunk = _delta_with_reasoning


_patch_langchain_openai_reasoning()


from .reasoning import effort_from_budget, model_uses_reasoning, openai_effort


def _enable_reasoning(
    model: str, base_url: str, reasoning_budget: int, overrides: dict[str, Any]
) -> None:
    """Apply the selected effort through each OpenAI-compatible API.

    GPT-5 uses Responses because Chat Completions cannot combine its reasoning
    with tools. OpenRouter takes an effort level, not a token count.
    """
    effort = effort_from_budget(reasoning_budget)
    native_openai = (
        base_url.rstrip("/") == "https://api.openai.com/v1"
        and "gpt-5" in model.lower()
        and "chat" not in model.lower()
    )
    if native_openai:
        extra_body = overrides.get("extra_body")
        has_reasoning_override = isinstance(extra_body, Mapping) and bool(
            extra_body.get("reasoning") or extra_body.get("reasoning_effort")
        )
        if (
            overrides.get("reasoning_effort") is None
            and not overrides.get("reasoning")
            and not has_reasoning_override
        ):
            applied = openai_effort(model, effort)
            overrides["reasoning"] = (
                {"effort": "none"}
                if applied == "none"
                else {"effort": applied, "summary": "auto"}
            )
        overrides.setdefault("use_responses_api", True)
    elif "openrouter.ai" in base_url:
        extra_body = dict(overrides.get("extra_body") or {})
        if "reasoning" not in extra_body and model_uses_reasoning(model):
            extra_body["reasoning"] = {"effort": openai_effort(model, effort)}
        if extra_body:
            overrides["extra_body"] = extra_body


def build_chat_model(
    *,
    base_url: str,
    model: str,
    api_key: str | None = None,
    timeout_s: float = 30.0,
    reasoning_budget: int = 8192,
    **overrides: Any,
) -> BaseChatModel:
    """Return a LangChain chat model pointed at an OpenAI-compatible API.

    The planner binds JSON tools at runtime. We do not pass
    ``response_format={"type": "json_object"}``; free-form JSON remains a
    fallback in :mod:`parser` when a model emits a plan instead of tools.
    """
    # Local import keeps planning_service importable without langchain
    # installed (e.g. for type-checking or partial unit tests).
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {
        "base_url": base_url,
        "model": model,
        "timeout": timeout_s,
    }
    # ChatOpenAI validates that a key exists even for local servers
    # (Ollama/vLLM) that need none — substitute a harmless placeholder.
    kwargs["api_key"] = api_key or "not-required"
    kwargs.update(overrides)
    _enable_reasoning(model, base_url, reasoning_budget, kwargs)
    return ChatOpenAI(**kwargs)
