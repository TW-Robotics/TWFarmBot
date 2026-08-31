"""Small Responses API adapter with native Programmatic Tool Calling.

This intentionally does not use LangChain's Chat Completions abstraction:
PTC is a Responses API feature and returns program/function-call items rather
than ``AIMessage.tool_calls``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Sequence

import httpx
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.runnables import Runnable


class OpenAIResponsesModel(Runnable[Any, AIMessage]):
    """Responses API model that continues native PTC programs."""

    model_name: str

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None,
        timeout_s: float,
    ) -> None:
        super().__init__()
        self.model_name = model
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key or ''}"},
            timeout=timeout_s,
        )
        self._tools: list[dict[str, Any]] = []
        self._invoke_tool: Callable[[str, dict[str, Any]], Any] | None = None

    def configure_tools(
        self,
        descriptors: Sequence[Any],
        invoke_tool: Callable[[str, dict[str, Any]], Any],
    ) -> None:
        """Attach application-owned tools after the registry is available."""
        self._invoke_tool = invoke_tool
        self._tools = [_descriptor_to_response_tool(d) for d in descriptors]
        self._tools.append({"type": "programmatic_tool_calling"})

    def invoke(self, input: Any, config: Any | None = None, **kwargs: Any) -> AIMessage:
        del config, kwargs
        final: AIMessage | None = None
        for event in self._iter_response(input):
            if event.get("type") == "final":
                final = event["message"]
        if final is None:
            raise RuntimeError("Responses API exceeded the tool continuation limit")
        return final

    def stream(self, input: Any, config: Any | None = None, **kwargs: Any):
        """Yield tool/program events as the hosted PTC loop progresses."""
        del config, kwargs
        for event in self._iter_response(input):
            kind = event.get("type")
            if kind in {"tool_call", "program"}:
                yield AIMessageChunk(
                    content="",
                    additional_kwargs={"stream_event": event},
                )
            elif kind == "final":
                message: AIMessage = event["message"]
                yield AIMessageChunk(
                    content=message.content,
                    additional_kwargs=message.additional_kwargs,
                )

    def _iter_response(self, input: Any):
        if not self._tools or self._invoke_tool is None:
            raise RuntimeError("Responses model tools are not configured")
        items = _messages_to_input(input)
        tool_log: list[dict[str, Any]] = []
        proposed_actions: list[dict[str, Any]] = []
        programs: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []
        for _ in range(100):
            request = {
                "model": self.model_name,
                "store": False,
                "input": items,
                "tools": self._tools,
            }
            # Force only the first turn. Continuation turns must be allowed to
            # return the approval summary instead of calling the same action.
            action_tool = _requested_action_tool(items) if not tool_log else None
            if action_tool:
                request["tool_choice"] = {"type": "function", "name": action_tool}
            response = self._client.post("/responses", json=request)
            if response.is_error:
                detail = response.text[:2000]
                raise RuntimeError(
                    f"Responses API returned HTTP {response.status_code}: {detail}"
                )
            payload = response.json()
            if payload.get("status") not in (None, "completed"):
                raise RuntimeError(f"Responses API ended with {payload.get('status')}")
            output = payload.get("output", [])
            trace.append({
                "turn": len(trace) + 1,
                "response_id": payload.get("id"),
                "status": payload.get("status"),
                "output": [_trace_item(item) for item in output],
            })
            items.extend(output)
            for event in _absorb_programs(output, programs):
                yield event
            calls = [item for item in output if item.get("type") == "function_call"]
            if calls:
                for call in calls:
                    args = json.loads(call.get("arguments", "{}"))
                    result = self._invoke_tool(str(call["name"]), args)
                    recorded = {
                        "name": str(call["name"]),
                        "args": args,
                        "result": result,
                        "caller": call.get("caller"),
                    }
                    tool_log.append(recorded)
                    if isinstance(result, dict) and result.get("status") == "proposed":
                        proposed_actions.append(
                            {
                                "kind": result.get("kind", call["name"]),
                                "params": result.get("params", args),
                            }
                        )
                    items.append(
                        {
                            "type": "function_call_output",
                            "call_id": call["call_id"],
                            "output": json.dumps(_compact_tool_result(result), default=str),
                            "caller": call.get("caller"),
                        }
                    )
                    yield {"type": "tool_call", **recorded}
                continue
            text = payload.get("output_text") or _message_text(output)
            text = _append_result_images(text, tool_log)
            if not tool_log and _requested_action_tool(items) == "move":
                args = _move_args_from_text(text)
                if args is not None:
                    result = self._invoke_tool("move", args)
                    recorded = {"name": "move", "args": args, "result": result, "caller": None}
                    tool_log.append(recorded)
                    if isinstance(result, dict) and result.get("status") == "proposed":
                        proposed_actions.append(
                            {"kind": result.get("kind", "move"), "params": result.get("params", args)}
                        )
                    yield {"type": "tool_call", **recorded}
            if text:
                yield {
                    "type": "final",
                    "message": AIMessage(
                        content=text,
                        additional_kwargs={
                            "tool_calls": tool_log,
                            "proposed_actions": proposed_actions,
                            "programs": programs,
                            "trace": trace,
                        },
                    ),
                }
                return
        raise RuntimeError("Responses API exceeded the tool continuation limit")


def _descriptor_to_response_tool(descriptor: Any) -> dict[str, Any]:
    schema = _strict_object_schema(descriptor.args_schema.model_json_schema())
    tool: dict[str, Any] = {
        "type": "function",
        "name": descriptor.name,
        "description": descriptor.policy.description or descriptor.name,
        "parameters": schema,
        "strict": True,
    }
    output_schema = getattr(descriptor, "output_schema", None)
    if output_schema:
        tool["output_schema"] = _normalize_output_schema(output_schema)
    if descriptor.is_introspection:
        tool["allowed_callers"] = ["programmatic"]
    return tool


def _trace_item(item: dict[str, Any]) -> dict[str, Any]:
    """Keep operational output items while excluding private reasoning."""
    fields = {
        "type",
        "id",
        "name",
        "call_id",
        "status",
        "arguments",
        "role",
        "code",
        "result",
        "caller",
    }
    result = {key: item[key] for key in fields if key in item}
    if item.get("type") == "message":
        result["text"] = _message_text([item])
    return result


def _compact_tool_result(value: Any) -> Any:
    """Avoid sending binary/base64 vision output back into the next LLM turn."""
    if isinstance(value, list):
        return [_compact_tool_result(item) for item in value]
    if isinstance(value, dict):
        return {
            key: (
                "[image available to the user]"
                if key in {"image_url", "image_urls"}
                and isinstance(item, (str, list))
                else _compact_tool_result(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, str) and value.startswith("data:image/"):
        return "[image available to the user]"
    return value


def _append_result_images(text: str, tool_log: list[dict[str, Any]]) -> str:
    """Expose final vision artifacts without dumping intermediate camera history."""
    artifacts: list[tuple[str, str]] = []
    labels = {
        "analyze_image": ("Similarity map",),
        "segment_image": ("Segmentation overlay", "Segmentation mask"),
        "visualize_image_features": (
            "PCA feature map",
            "PCA component map",
            "PCA cluster map",
        ),
        "estimate_traversability": ("Traversability map",),
    }
    for call in tool_log:
        name = str(call.get("name", ""))
        if name not in labels:
            continue
        result = call.get("result", {})
        if not isinstance(result, dict):
            continue
        images: list[str] = []
        value = result.get("image_url")
        if isinstance(value, str):
            images.append(value)
        values = result.get("image_urls")
        if isinstance(values, list):
            images.extend(value for value in values if isinstance(value, str))
        for index, url in enumerate(images):
            label_set = labels[name]
            label = label_set[index] if index < len(label_set) else "Analysis result"
            artifacts.append((label, url))

    # A plain photo request is normally verified with get_images. Show only the
    # newest frame from the final lookup, not every historical frame fetched by
    # intermediate agent turns.
    if not artifacts:
        for call in reversed(tool_log):
            if call.get("name") != "get_images":
                continue
            result = call.get("result", {})
            images = result.get("images", []) if isinstance(result, dict) else []
            if images and isinstance(images[0], dict):
                url = images[0].get("attachment_url")
                if isinstance(url, str):
                    artifacts.append(("FarmBot photo", url))
            break

    unique = dict.fromkeys(artifacts)
    additions = [f"![{label}]({url})" for label, url in unique if url not in text]
    return text + ("\n\n" + "\n\n".join(additions) if additions else "")


def _strict_object_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Normalize Pydantic JSON Schema to OpenAI strict function schema rules."""
    def visit(value: Any) -> Any:
        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, dict):
            return value
        normalized = {key: visit(item) for key, item in value.items()}
        if normalized.get("type") == "object" or "properties" in normalized:
            properties = normalized.setdefault("properties", {})
            normalized["additionalProperties"] = False
            normalized["required"] = list(properties)
        return normalized

    normalized = visit(schema)
    normalized.setdefault("type", "object")
    normalized.setdefault("properties", {})
    normalized["additionalProperties"] = False
    normalized["required"] = list(normalized["properties"])
    return normalized


def _normalize_output_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Keep map objects (additionalProperties: schema) intact for PTC."""

    def visit(value: Any) -> Any:
        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, dict):
            return value
        normalized = {key: visit(item) for key, item in value.items()}
        if normalized.get("type") == "object" or "properties" in normalized:
            properties = normalized.setdefault("properties", {})
            extra = normalized.get("additionalProperties")
            if not isinstance(extra, dict):
                normalized["additionalProperties"] = False
            normalized.setdefault("required", list(properties))
        return normalized

    return visit(schema)


def _absorb_programs(
    output: Sequence[dict[str, Any]], programs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    by_id = {item.get("call_id"): item for item in programs}

    def upsert(call_id: Any, **fields: Any) -> dict[str, Any]:
        current = by_id.get(call_id)
        if current is None:
            current = {"call_id": call_id, "code": "", "status": "running", "result": None}
            programs.append(current)
            by_id[call_id] = current
        current.update({key: value for key, value in fields.items() if value is not None})
        return current

    for item in output:
        kind = item.get("type")
        if kind == "program":
            recorded = upsert(
                item.get("call_id"),
                code=item.get("code") or "",
                status="running",
            )
            events.append({"type": "program", **recorded})
        elif kind == "program_output":
            recorded = upsert(
                item.get("call_id"),
                result=item.get("result"),
                status=item.get("status") or "completed",
            )
            events.append({"type": "program", **recorded})
    return events


def _messages_to_input(messages: Sequence[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for message in messages:
        role = getattr(message, "type", "user")
        role = {"human": "user", "ai": "assistant"}.get(role, role)
        content = _compact_input_text(str(getattr(message, "content", "") or ""))
        out.append({"role": role, "content": content})
    return out


def _compact_input_text(text: str) -> str:
    """Strip persisted base64 images from old browser conversation history."""
    return re.sub(
        r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+",
        "[previous image available to the user]",
        text,
    )


def _message_text(output: Sequence[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in output:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                parts.append(str(content.get("text", "")))
    return "".join(parts)


def _requested_action_tool(items: Sequence[dict[str, Any]]) -> str | None:
    """Return the requested physical tool, if the user clearly names one."""
    user_messages = [
        str(item.get("content", ""))
        for item in items
        if item.get("role") == "user"
    ]
    user_text = user_messages[-1] if user_messages else ""
    match = re.search(
        r"\b(inspect|move|water|home|mount|dismount|write\s+(?:to\s+)?pin)\b",
        user_text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return {
        "inspect": "inspect_zone",
        "move": "move",
        "water": "water",
        "home": "find_home",
        "mount": "mount_tool",
        "dismount": "dismount_tool",
        "write": "write_pin",
    }[match.group(1).lower().split()[0]]


def _move_args_from_text(text: str) -> dict[str, float] | None:
    """Recover an explicit XYZ proposal when a tool-capable model answers text-only."""
    values: dict[str, float] = {}
    for axis in ("x", "y", "z"):
        match = re.search(rf"\b{axis}\s*[=:]?\s*(-?\d+(?:\.\d+)?)", text, re.IGNORECASE)
        if match:
            values[axis] = float(match.group(1))
    if len(values) == 3:
        return values
    tuple_match = re.search(
        r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)",
        text,
    )
    if tuple_match:
        return dict(zip(("x", "y", "z"), (float(v) for v in tuple_match.groups())))
    return None
