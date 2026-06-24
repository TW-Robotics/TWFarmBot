"""LangChain chat-model factory for the planner.

We use ``langchain_openai.ChatOpenAI`` for both OpenRouter and any
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
