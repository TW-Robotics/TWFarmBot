"""Agent loop on Pydantic AI.

Pydantic AI owns the turn loop: tool-call parsing and validation, retries,
message history, streaming, usage limits, deferred tool search and human
approval. This module only adds the farm-specific pieces:

- tools built from ``ToolRegistry`` (safety/approval gate, motion rules),
- the goal checklist and "keep going" nudge,
- operator cancel and mid-run follow-ups (``cancel.py``),
- translation of Pydantic AI events into the SSE dicts the UI consumes.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from pydantic import BaseModel, ValidationError
from pydantic_ai import (
    Agent,
    ApprovalRequired,
    BinaryContent,
    DeferredToolRequests,
    DeferredToolResults,
    ModelRetry,
    RunContext,
    Tool,
    ToolDenied,
    ToolReturn,
)
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ThinkingPart,
    ThinkingPartDelta,
    UserPromptPart,
)
from pydantic_ai.run import AgentRunResultEvent
from pydantic_ai.usage import UsageLimits

from ..tool_results import (
    _provider_image_url,
    _sse_images,
    compact_input_text,
    compact_tool_result,
)
from .approval_gate import ApprovalGate
from .cancel import (
    STOP_CANCELLED,
    RunCancelled,
    begin_run,
    drain_followup,
    end_run,
    is_cancelled,
    peek_followup,
)
from .context_builder import ContextBuilder
from .facts import record_tool_fact
from .goal import apply_goal_update, format_goal
from .metrics import Metrics
from .tool_policy import ToolCategory, ToolDescriptor
from .tool_registry import CORE_TOOLS, ToolRegistry
from .tracing import (
    call_url,
    finish_agent_call,
    is_enabled,
    log_span,
    log_tool_call,
    start_agent_call,
)

_log = logging.getLogger(__name__)

STOP_DONE = "done"
STOP_MAX_TURNS = "max_turns"
STOP_MAX_ERRORS = "max_errors"
MAX_GOAL_NUDGES = 2
JOG_APPROVAL_MM = 50.0
_EMPTY_TEXT = "I could not produce a response. Try again with a more specific request."
_MAX_SUMMARY_LINES = 8
_MOTION_TOOLS = frozenset({"move", "move_axis", "move_path", "goto_named", "run_script"})
_LIVE_POSE_TOOLS = _MOTION_TOOLS | {"find_home", "scan_ndre", "capture_ndre", "get_position"}
# Tools Pydantic AI provides itself (`search_tools`) or that the loop handles.
_HIDDEN_TOOLS = frozenset({"tool_search"})

# Runs paused for approval, kept so the next request can resume them.
_PAUSED: dict[str, tuple[list[Any], "RunState"]] = {}


@dataclass(frozen=True)
class AgentTurnResult:
    """Result of one agent run."""

    response: str
    thinking: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    proposed_actions: list[dict[str, Any]] = field(default_factory=list)
    programs: list[dict[str, Any]] = field(default_factory=list)
    metrics: Metrics = field(default_factory=Metrics)
    stop_reason: str = STOP_DONE


class _TooManyErrors(Exception):
    """Raised from a tool after ``max_errors`` failed calls in a row."""


@dataclass
class RunState:
    """Per-run state handed to every tool as ``ctx.deps``."""

    loop: "AgentLoop"
    thread_id: str
    streaming: bool
    planner: bool = False
    weave_call: Any = None
    goal: dict[str, Any] = field(default_factory=dict)
    tool_log: list[dict[str, Any]] = field(default_factory=list)
    proposed: list[dict[str, Any]] = field(default_factory=list)
    results: dict[str, Any] = field(default_factory=dict)  # call id -> raw result
    followups: list[str] = field(default_factory=list)
    errors: int = 0
    nudges: int = 0
    saw_action: bool = False
    motion_approved: bool = False
    resumed: bool = False
    force_motion: bool = False  # this response batches several moves or a new plan
    approved: set[str] = field(default_factory=set)  # call ids the operator OK'd


# ───────────────────────────── tool execution ────────────────────────────────


def _plain(value: Any) -> Any:
    """Recursively convert Pydantic models to plain dicts."""
    if isinstance(value, BaseModel):
        return _plain(value.model_dump())
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


def _succeeded(result: Any) -> bool:
    return not (
        isinstance(result, dict)
        and (result.get("status") == "error" or result.get("error"))
    )


def _current_xyz() -> list[float] | None:
    try:
        from farmbot_gateway import farmbot

        xyz = farmbot.backend.get_xyz()
        return [float(xyz["x"]), float(xyz["y"]), float(xyz["z"])]
    except Exception:  # noqa: BLE001
        return None


def _travel_mm(target: dict[str, Any]) -> float:
    try:
        goal = [float(target[axis]) for axis in ("x", "y", "z")]
    except (KeyError, TypeError, ValueError):
        return JOG_APPROVAL_MM + 1
    here = _current_xyz()
    if here is None:
        return JOG_APPROVAL_MM + 1
    return sum((goal[i] - here[i]) ** 2 for i in range(3)) ** 0.5


def _goto_named_travel_mm(args: dict[str, Any]) -> float:
    try:
        from spatial_service import resolve_named

        target = resolve_named(str(args.get("name") or ""))
        return _travel_mm({"x": target["x"], "y": target["y"], "z": args.get("z", target["z"])})
    except Exception:  # noqa: BLE001
        return JOG_APPROVAL_MM + 1


def _is_new_plan(args: dict[str, Any]) -> bool:
    remaining = args.get("remaining")
    return bool(str(args.get("objective") or "").strip()) or (
        isinstance(remaining, list) and any(str(step).strip() for step in remaining)
    )


def _needs_approval(st: RunState, d: ToolDescriptor, args: dict[str, Any]) -> bool:
    """Risky acts pause for the operator; jogs under 50 mm run immediately."""
    if st.loop.skip_approval or not st.streaming:
        return False
    name = d.name
    if name == "set_goal":
        return _is_new_plan(args)
    if st.motion_approved:
        return False
    if name in _MOTION_TOOLS and st.force_motion:
        return True
    if d.policy.requires_approval:
        return True
    if name == "move_axis":
        try:
            return abs(float(args.get("distance") or 0)) >= JOG_APPROVAL_MM
        except (TypeError, ValueError):
            return True
    if name == "move":
        return bool(st.goal) or _travel_mm(args) >= JOG_APPROVAL_MM
    if name == "goto_named":
        return _goto_named_travel_mm(args) >= JOG_APPROVAL_MM
    return False


def _dispatch(st: RunState, d: ToolDescriptor, args: dict[str, Any]) -> dict[str, Any]:
    if d.name == "set_goal":
        st.goal = apply_goal_update(st.goal, args)
        return {"status": "ok", "goal": st.goal}
    if d.is_introspection:
        if d.execute is None:
            return {"error": f"introspection tool {d.name!r} has no executor"}
        return d.execute(args)
    res = st.loop.gate.resolve(
        d, args, propose_only=st.loop.propose_only, allow_actions=st.loop.allow_actions
    )
    out: dict[str, Any] = {"status": res.status, "kind": res.kind, "params": res.params}
    if res.note:
        out["note"] = res.note
    if res.error:
        out["error"] = res.error
    return out


def _execute(ctx: RunContext[RunState], d: ToolDescriptor, kwargs: dict[str, Any]) -> Any:
    st = ctx.deps
    name = d.name
    try:
        d.args_schema.model_validate(kwargs)
    except ValidationError as err:
        raise ModelRetry(f"Invalid arguments for {name}: {err.errors(include_url=False)}")
    args = _plain(kwargs)
    if _needs_approval(st, d, args) and not ctx.tool_call_approved:
        raise ApprovalRequired
    if ctx.tool_call_approved:
        st.motion_approved = True
    if d.policy.category == ToolCategory.ACT:
        st.saw_action = True

    start = time.perf_counter()
    xyz = _current_xyz()
    if peek_followup():
        result: Any = {"status": "skipped", "kind": name, "params": args, "note": "skipped: new user instruction"}
    elif is_cancelled():
        result = {"status": "error", "kind": name, "params": args, "error": "stopped by operator"}
    else:
        try:
            result = _plain(_dispatch(st, d, args))
        except Exception as err:  # noqa: BLE001
            result = {"error": f"{type(err).__name__}: {err}"}
        record_tool_fact(name, args, result, xyz=xyz, store=getattr(st.loop.registry, "_system_state", None))
        if name in _MOTION_TOOLS and _succeeded(result):
            st.motion_approved = True
    if is_enabled():
        log_tool_call(name, args, compact_tool_result(result), latency_s=time.perf_counter() - start, parent=st.weave_call)
    if isinstance(result, dict) and result.get("_resireg_latency_s"):
        st.loop.resireg_s += float(result["_resireg_latency_s"])

    st.tool_log.append({"name": name, "args": args, "result": result})
    st.results[ctx.tool_call_id] = result
    if isinstance(result, dict) and result.get("status") == "proposed":
        st.proposed.append({"kind": result.get("kind", name), "params": result.get("params", args)})
    st.errors = 0 if _succeeded(result) else st.errors + 1
    if st.errors >= st.loop.max_errors:
        raise _TooManyErrors

    image = _provider_image_url(name, result)
    content = [BinaryContent.from_data_uri(image)] if image else None
    return ToolReturn(return_value=compact_tool_result(result), content=content)


def _build_tool(d: ToolDescriptor, core: frozenset[str]) -> Tool[RunState]:
    def run(ctx: RunContext[RunState], **kwargs: Any) -> Any:
        return _execute(ctx, d, kwargs)

    tool = Tool.from_schema(
        run,
        name=d.name,
        description=d.policy.description or d.name,
        json_schema=d.args_schema.model_json_schema(),
        takes_ctx=True,
        sequential=d.policy.category == ToolCategory.ACT,  # never overlap motion
    )
    tool.defer_loading = d.name not in core  # found via search_tools
    return tool


# ─────────────────────────── agent construction ──────────────────────────────


class _FarmRules(AbstractCapability[RunState]):
    """Per-step rules: force `think` first/after errors; gate batched motion."""

    @staticmethod
    def _force_think(ctx: RunContext[RunState]) -> bool:
        if not ctx.deps.loop.force_think or "think" not in ctx.deps.loop.registry.by_name():
            return False
        return (ctx.run_step <= 1 and not ctx.deps.resumed) or ctx.deps.errors > 0

    def get_model_settings(self) -> Callable[[RunContext[RunState]], dict[str, Any]]:
        # `required` with only `think` on offer: local OpenAI-compatible servers
        # honor that far more reliably than a named choice among many tools.
        return lambda ctx: {"tool_choice": "required"} if self._force_think(ctx) else {}

    async def prepare_tools(self, ctx: RunContext[RunState], tool_defs: list[Any]) -> list[Any]:
        if not self._force_think(ctx):
            return tool_defs
        return [t for t in tool_defs if t.name == "think"] or tool_defs

    async def before_model_request(self, ctx: RunContext[RunState], request_context: Any) -> Any:
        """Operator text sent mid-run reaches the model once the tool batch is done."""
        followup = drain_followup(ctx.deps.thread_id)
        if is_cancelled(ctx.deps.thread_id):
            return request_context
        if followup:
            ctx.enqueue(followup)
            ctx.deps.followups.append(followup)
        return request_context

    async def after_model_request(self, ctx: RunContext[RunState], *, request_context: Any, response: ModelResponse) -> ModelResponse:
        """Several moves in one response, or a new plan, need the operator's OK."""
        st = ctx.deps
        calls = [p for p in response.parts if isinstance(p, ToolCallPart)]
        new_plan = any(c.tool_name == "set_goal" and _is_new_plan(c.args_as_dict()) for c in calls)
        if new_plan:
            st.motion_approved = False
        st.force_motion = not st.motion_approved and (
            new_plan or sum(c.tool_name in _MOTION_TOOLS for c in calls) > 1
        )
        return response


def _history(messages: list[dict[str, Any]], include_reasoning: bool) -> list[Any]:
    out: list[Any] = []
    for msg in messages:
        text = compact_input_text(str(msg.get("content", "") or ""))
        role = msg.get("role")
        if role == "user":
            out.append(ModelRequest(parts=[UserPromptPart(content=text)]))
        elif role == "assistant":
            parts: list[Any] = []
            if include_reasoning and msg.get("thinking"):
                parts.append(ThinkingPart(content=str(msg["thinking"])))
            parts.append(TextPart(content=text))
            out.append(ModelResponse(parts=parts))
    return out


def _closing_summary(tool_log: list[dict[str, Any]], stop_reason: str) -> str:
    """User-facing message when the model ends without text."""
    lines: list[str] = []
    if stop_reason == STOP_MAX_TURNS:
        lines.append("I hit the step limit before finishing.")
    elif stop_reason == STOP_MAX_ERRORS:
        lines.append("I stopped after repeated tool errors.")
    elif stop_reason == STOP_CANCELLED:
        lines.append("Stopped by the operator.")
    for record in tool_log[-_MAX_SUMMARY_LINES:]:
        name = str(record.get("name", "tool"))
        result = record.get("result")
        error = result.get("error") if isinstance(result, dict) else None
        status = result.get("status") if isinstance(result, dict) else None
        if error:
            lines.append(f"- {name} failed: {error}")
        elif status == "proposed":
            lines.append(f"- {name} proposed, awaiting approval.")
        else:
            lines.append(f"- {name}: done.")
    return "\n".join(lines) if lines else _EMPTY_TEXT


def _iter_sync(make_cm: Callable[[], Any]) -> Iterator[Any]:
    """Drive Pydantic AI's async event stream from a synchronous generator."""
    loop = asyncio.new_event_loop()
    cm = make_cm()
    events = loop.run_until_complete(cm.__aenter__())
    try:
        while True:
            try:
                yield loop.run_until_complete(events.__anext__())
            except StopAsyncIteration:
                return
    finally:
        loop.run_until_complete(cm.__aexit__(None, None, None))
        loop.close()


# ──────────────────────────────── AgentLoop ──────────────────────────────────


class AgentLoop:
    """Multi-turn tool-calling loop (Pydantic AI-backed)."""

    def __init__(
        self,
        model: Any,
        tool_registry: ToolRegistry,
        approval_gate: ApprovalGate,
        context_builder: ContextBuilder,
        *,
        model_name: str = "unknown",
        propose_only: bool = False,
        allow_actions: bool = True,
        include_reasoning: bool = False,
        max_turns: int = 100,
        max_consecutive_errors: int = 5,
        skip_approval: bool = False,
        core_tools: frozenset[str] = CORE_TOOLS,
        force_think: bool = False,
    ) -> None:
        self.registry = tool_registry
        self.gate = approval_gate
        self.propose_only = propose_only
        self.allow_actions = allow_actions
        self.skip_approval = skip_approval
        self.force_think = force_think
        self.max_errors = max_consecutive_errors
        self.resireg_s = 0.0
        self._model = model
        self._context = context_builder
        self._model_name = model_name
        self._include_reasoning = include_reasoning
        self._max_turns = max_turns
        descriptors = [d for d in tool_registry.descriptors() if d.name not in _HIDDEN_TOOLS]
        self._tools = [_build_tool(d, core_tools) for d in descriptors]
        self._agents: dict[bool, Agent[RunState, Any]] = {}

    def _agent(self, planner: bool) -> Agent[RunState, Any]:
        if planner not in self._agents:
            prompt = (
                self._context.planner_system_prompt()
                if planner
                else self._context.chat_system_prompt(voice=self.skip_approval)
            )
            agent: Agent[RunState, Any] = Agent(
                self._model,
                deps_type=RunState,
                output_type=[str, DeferredToolRequests],
                instructions=prompt,
                tools=self._tools,
                capabilities=[_FarmRules()],
                retries={"tools": 2, "output": MAX_GOAL_NUDGES},
            )

            @agent.instructions
            def goal_note(ctx: RunContext[RunState]) -> str:
                return format_goal(ctx.deps.goal)

            @agent.output_validator
            def keep_working(ctx: RunContext[RunState], output: Any) -> Any:
                st = ctx.deps
                if isinstance(output, str) and st.goal.get("remaining") and st.nudges < MAX_GOAL_NUDGES:
                    st.nudges += 1
                    raise ModelRetry(
                        "[continue] The goal still has unfinished steps: "
                        + "; ".join(st.goal["remaining"])
                        + ". Do the next step now with tool calls. If you are truly blocked, "
                        "call set_goal with remaining=[] and tell the operator exactly what you need."
                    )
                return output

            self._agents[planner] = agent
        return self._agents[planner]

    # ── event translation ────────────────────────────────────────────────────

    def _translate(self, ev: Any, st: RunState, calls: dict[str, dict[str, Any]]) -> Iterator[dict[str, Any]]:
        if isinstance(ev, PartStartEvent):
            if isinstance(ev.part, TextPart) and ev.part.content:
                yield {"type": "delta", "content": ev.part.content}
            elif isinstance(ev.part, ThinkingPart) and ev.part.content:
                yield {"type": "thinking", "content": ev.part.content}
        elif isinstance(ev, PartDeltaEvent):
            if isinstance(ev.delta, TextPartDelta) and ev.delta.content_delta:
                yield {"type": "delta", "content": ev.delta.content_delta}
            elif isinstance(ev.delta, ThinkingPartDelta) and ev.delta.content_delta:
                yield {"type": "thinking", "content": ev.delta.content_delta}
        elif isinstance(ev, FunctionToolCallEvent):
            call = {"id": ev.part.tool_call_id, "name": ev.part.tool_name, "args": ev.part.args_as_dict()}
            calls[call["id"]] = call
            d = self.registry.by_name().get(call["name"])
            paused = d and call["id"] not in st.approved and _needs_approval(st, d, call["args"])
            if not paused:  # a paused call shows as an approval card instead
                yield {"type": "tool_start", **call}
        elif isinstance(ev, FunctionToolResultEvent):
            call = calls.get(ev.tool_call_id, {"id": ev.tool_call_id, "name": ev.part.tool_name, "args": {}})
            result = st.results.pop(ev.tool_call_id, None)
            if result is None:  # validation retry: nothing ran
                result = {"status": "error", "error": str(getattr(ev.part, "content", ""))}
            yield {
                "type": "tool_call",
                **call,
                "result": compact_tool_result(result),
                "images": _sse_images(call["name"], result),
            }
            if call["name"] == "think":
                yield {"type": "thinking", "content": str(call["args"].get("thought", ""))}
            elif call["name"] == "set_goal":
                yield {"type": "goal", "goal": st.goal}
            if call["name"] in _LIVE_POSE_TOOLS and _succeeded(result):
                xyz = _current_xyz()
                if xyz:
                    yield {"type": "position", "xyz": dict(zip("xyz", xyz))}

    def _drive(
        self,
        st: RunState,
        *,
        prompt: str | None,
        history: list[Any],
        results: DeferredToolResults | None = None,
        stop_after_propose: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Run the agent, yielding SSE events; the last one is ``_end``."""
        agent = self._agent(st.planner)
        calls: dict[str, dict[str, Any]] = {}
        stop = STOP_DONE
        run_result = None
        try:
            for ev in _iter_sync(
                lambda: agent.run_stream_events(
                    prompt,
                    message_history=history,
                    deferred_tool_results=results,
                    deps=st,
                    usage_limits=UsageLimits(request_limit=self._max_turns),
                )
            ):
                if is_cancelled(st.thread_id):
                    raise RunCancelled("stopped by operator")
                if isinstance(ev, AgentRunResultEvent):
                    run_result = ev.result
                    continue
                yield from self._translate(ev, st, calls)
                while st.followups:
                    yield {"type": "followup", "text": st.followups.pop(0)}
                if stop_after_propose and st.saw_action and isinstance(ev, PartStartEvent):
                    break  # the batch with the proposals is done; stop before the next reply
        except UsageLimitExceeded:
            stop = STOP_MAX_TURNS
        except _TooManyErrors:
            stop = STOP_MAX_ERRORS
        except UnexpectedModelBehavior as err:
            # Empty replies exhaust the output retries: end quietly with a summary.
            # Anything else is a tool the model kept getting wrong.
            stop = STOP_DONE if "output retries" in str(err) else STOP_MAX_ERRORS
        yield {"type": "_end", "stop": stop, "result": run_result}

    # ── public API ───────────────────────────────────────────────────────────

    def _finish(self, st: RunState, end: dict[str, Any], started: float) -> tuple[str, str, Metrics]:
        run_result = end["result"]
        text = run_result.output if run_result and isinstance(run_result.output, str) else ""
        stop = STOP_CANCELLED if is_cancelled(st.thread_id) else end["stop"]
        metrics = Metrics(total_latency_s=time.perf_counter() - started, resireg_latency_s=self.resireg_s)
        if run_result:
            usage = run_result.usage
            metrics.prompt_tokens, metrics.completion_tokens = usage.input_tokens, usage.output_tokens
            metrics.total_tokens = usage.input_tokens + usage.output_tokens
            metrics.turn_count = usage.requests
            metrics.llm_latency_s = metrics.total_latency_s
            metrics._recalc_tokens_per_s()
        return text, stop, metrics

    def _collect(self, st: RunState, prompt: str | None, history: list[Any], *, stop_after_propose: bool) -> AgentTurnResult:
        started = time.perf_counter()
        thinking: list[str] = []
        for ev in self._drive(st, prompt=prompt, history=history, stop_after_propose=stop_after_propose):
            if ev["type"] == "thinking":
                thinking.append(ev["content"])
            elif ev["type"] == "_end":
                text, stop, metrics = self._finish(st, ev, started)
        return AgentTurnResult(
            response=text or _closing_summary(st.tool_log, stop),
            thinking="".join(thinking).strip() or None,
            tool_calls=st.tool_log,
            proposed_actions=[] if st.planner else st.proposed,
            metrics=metrics,
            stop_reason=stop,
        )

    def _split(self, messages: list[dict[str, Any]]) -> tuple[str | None, list[Any]]:
        history = _history(messages, self._include_reasoning)
        if messages and messages[-1].get("role") == "user":
            return compact_input_text(str(messages[-1].get("content", "") or "")), history[:-1]
        return None, history

    def run(self, messages: list[dict[str, Any]]) -> AgentTurnResult:
        """Run the loop synchronously and return the final result."""
        st = RunState(self, uuid.uuid4().hex, streaming=False)
        prompt, history = self._split(messages)
        return self._collect(st, prompt, history, stop_after_propose=False)

    def plan_request(self, request: str) -> AgentTurnResult:
        """Planner-mode loop: gather introspection, collect action proposals."""
        st = RunState(self, uuid.uuid4().hex, streaming=False, planner=True)
        return self._collect(st, request, [], stop_after_propose=True)

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        thread_id: str | None = None,
        resume: dict[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Run the loop and yield SSE-style events."""
        started = time.perf_counter()
        thread_id = thread_id or uuid.uuid4().hex
        prompt, history = self._split(messages)
        call = start_agent_call(thread_id=thread_id, prompt=prompt or "")
        weave_url = call_url(call)
        begin_run(thread_id)
        output: dict[str, Any] = {}
        try:
            yield {"type": "thread", "thread_id": thread_id, "weave_url": weave_url}
            results = None
            if resume is not None and thread_id in _PAUSED:
                history, st = _PAUSED.pop(thread_id)
                approved = set(resume.get("approved_ids", []))
                pending = st.results.pop("_pending", [])
                results = DeferredToolResults(
                    approvals={i: (True if i in approved else ToolDenied("rejected by approver")) for i in pending}
                )
                st.resumed, st.weave_call = True, call
                st.approved = approved
                prompt = None
            else:
                st = RunState(self, thread_id, streaming=True, weave_call=call)
            thinking: list[str] = []
            response: list[str] = []
            for ev in self._drive(st, prompt=prompt, history=history, results=results):
                kind = ev["type"]
                if kind != "_end":
                    if kind == "thinking":
                        thinking.append(ev["content"])
                        log_span("twfarmbot.thinking", {}, ev["content"], display_name="thinking", parent=call)
                    elif kind == "delta":
                        response.append(ev["content"])
                    yield ev
                    continue
                run_result = ev["result"]
                if run_result and isinstance(run_result.output, DeferredToolRequests):
                    ids = [c.tool_call_id for c in run_result.output.approvals]
                    st.results["_pending"] = ids
                    _PAUSED[thread_id] = (run_result.all_messages(), st)
                    output = {"stop_reason": "approval"}
                    yield {
                        "type": "approval",
                        "thread_id": thread_id,
                        "pending_approvals": [
                            {"name": c.tool_name, "args": c.args_as_dict(), "id": c.tool_call_id}
                            for c in run_result.output.approvals
                        ],
                    }
                    return
                text, stop, metrics = self._finish(st, ev, started)
                if not text and not response:
                    text = _closing_summary(st.tool_log, stop)
                    yield {"type": "delta", "content": text}
                elif stop == STOP_CANCELLED:
                    yield {"type": "delta", "content": "Stopped by the operator."}
                meta = {
                    "type": "meta",
                    "thread_id": thread_id,
                    "tool_calls": [
                        {
                            "name": r["name"],
                            "args": r["args"],
                            "result": compact_tool_result(r["result"]),
                            "images": _sse_images(r["name"], r["result"]),
                        }
                        for r in st.tool_log
                    ],
                    "proposed_actions": st.proposed,
                    "programs": [],
                    "metrics": metrics.to_dict(),
                    "stop_reason": stop,
                    "goal": dict(st.goal),
                    "weave_url": weave_url,
                }
                output = {
                    "thinking": "".join(thinking).strip(),
                    "response": "".join(response).strip(),
                    "tools": meta["tool_calls"],
                    "stop_reason": stop,
                    "metrics": meta["metrics"],
                }
                yield meta
        except RunCancelled:
            output = {"stop_reason": STOP_CANCELLED}
            yield {"type": "delta", "content": "Stopped by the operator."}
            yield {
                "type": "meta", "thread_id": thread_id, "tool_calls": [], "proposed_actions": [],
                "programs": [], "metrics": Metrics().to_dict(), "stop_reason": STOP_CANCELLED,
                "weave_url": weave_url,
            }
        except Exception as err:
            finish_agent_call(call, output=output or None, exception=err)
            raise
        else:
            finish_agent_call(call, output=output or None)
        finally:
            end_run(thread_id)
