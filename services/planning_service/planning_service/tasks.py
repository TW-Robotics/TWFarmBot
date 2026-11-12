"""Long-horizon tasks: plan once, approve once, run step by step in the background.

- A planner ``Agent`` (structured output) splits the objective into short steps.
- The operator approves the whole plan once; steps then run without per-move
  pauses (the safety box and e-stop still apply to every motion).
- Each step is its own short ``AgentLoop`` run with a fresh, small context, so
  a small model only has to do one thing at a time. Every step writes a normal
  trajectory (thread id ``<task>-s<step>-a<attempt>``).
- A verifier ``Agent`` judges each step from its tool calls. A failed step is
  retried, then the rest is re-planned, and only then does the task fail.
- State is saved after every step, so a restart resumes at the current step.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from .harness import AgentLoop
from .harness.cancel import RunCancelled, begin_run, cancel_run, end_run
from .tool_results import compact_tool_result

_log = logging.getLogger(__name__)

# Calls that only look things up: a step made of these did not do its action.
_LOOKUPS = frozenset(
    {"think", "get_help", "tool_search", "set_goal", "get_status", "get_position",
     "get_positions", "get_health", "get_messages", "recall_memories"}
)
# Steps that need a grasping tool fail up front while no end effector is configured.
_GRASP_STEP = re.compile(r"\b(?:pick|grab|grasp|release|let go|drop|place|put down)\b", re.IGNORECASE)
_GRASP_TOOLS = frozenset({"end_effector"})

MAX_ATTEMPTS = 2
MAX_REPLANS = 2
ACTIVE = ("planning", "running")


class Plan(BaseModel):
    steps: list[str] = Field(
        description="Ordered, short, imperative robot steps, e.g. 'Go to Carrot'. One action each."
    )


class StepVerdict(BaseModel):
    ok: bool = Field(description="True only if the tool results show the step really succeeded.")
    evidence: str = Field(description="One sentence citing the tool results.")


_PLANNER = """You plan multi-step jobs for a farm gantry robot (FarmBot).
Split the objective into the fewest short, imperative steps; each step is one
robot action an operator agent can do with its tools (go to a named place or
coordinates, mount/dismount a tool, take a photo, scan, water, pick, place...).
Do not add steps that only look things up; the step agent does that itself.
If steps are already done, plan only what remains."""

_VERIFIER = """You check whether one step of a robot job succeeded.
Judge only from the tool calls and their results, not from the agent's claims.
No tool call, or an error result, means the step did not succeed.
The step's action itself must have been done by a tool that does it (a move,
a capture, water, a pick or release tool...). Only reading status or position,
or the agent saying the action was not needed, does NOT complete an action
step. Moving next to an object does not pick it up or release it."""

_STEP = """You are doing step {index} of {total} of a longer job.
Objective: {objective}
Done so far: {done}
Current step: {step}

Do only the current step with tool calls, then verify it (e.g. get_position
after moving, capture to look). Finish with one sentence: what you did and the
evidence. If it cannot be done, say exactly why."""


_RUNNER: "TaskRunner | None" = None


def set_runner(runner: "TaskRunner | None") -> None:
    """Make ``runner`` the one chat hands approved plans to."""
    global _RUNNER
    _RUNNER = runner


def current_runner() -> "TaskRunner | None":
    return _RUNNER


def tasks_dir() -> Path:
    return Path(os.getenv("TWFB_TASKS_DIR") or Path.cwd() / "data" / "tasks")


class TaskStore:
    """One JSON file per task; written atomically after every change."""

    def __init__(self, directory: Path | None = None) -> None:
        self.dir = directory or tasks_dir()
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, task: dict[str, Any]) -> None:
        task["updated"] = time.time()
        tmp = self.dir / f"{task['id']}.json.tmp"
        tmp.write_text(json.dumps(task, indent=2))
        tmp.replace(self.dir / f"{task['id']}.json")

    def get(self, task_id: str) -> dict[str, Any] | None:
        path = self.dir / f"{Path(task_id).name}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def list(self) -> list[dict[str, Any]]:
        tasks = [json.loads(p.read_text()) for p in self.dir.glob("*.json")]
        return sorted(tasks, key=lambda t: t.get("created", 0), reverse=True)


def _steps(texts: list[str]) -> list[dict[str, Any]]:
    return [
        {"text": t.strip(), "status": "pending", "attempts": 0, "evidence": "", "threads": []}
        for t in texts
        if t.strip()
    ]


class TaskRunner:
    """Plans, runs and resumes tasks. One task drives the robot at a time."""

    def __init__(self, make_loop: Callable[[], AgentLoop], model: Any, store: TaskStore | None = None) -> None:
        self._make_loop = make_loop
        self._store = store or TaskStore()
        self._planner = Agent(model, output_type=Plan, instructions=_PLANNER)
        self._verifier = Agent(model, output_type=StepVerdict, instructions=_VERIFIER)
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._current_thread: str | None = None
        self._active: str | None = None
        self._cancel: set[str] = set()

    # ── planning ─────────────────────────────────────────────────────────────

    def _plan(self, objective: str, done: list[str], failure: str = "") -> list[str]:
        prompt = f"Objective: {objective}"
        if done:
            prompt += "\nAlready done: " + "; ".join(done)
        if failure:
            prompt += f"\nThe last attempt failed: {failure}\nPlan a different way to finish."
        return self._planner.run_sync(prompt).output.steps

    def create(self, objective: str) -> dict[str, Any]:
        task = {
            "id": uuid.uuid4().hex[:12],
            "objective": objective.strip(),
            "status": "planning",
            "steps": [],
            "replans": 0,
            "error": "",
            "created": time.time(),
        }
        self._store.save(task)
        try:
            task["steps"] = _steps(self._plan(task["objective"], []))
            task["status"] = "awaiting_approval" if task["steps"] else "failed"
            task["error"] = "" if task["steps"] else "The planner returned no steps."
        except Exception as err:  # noqa: BLE001 - surface planner errors to the UI
            task["status"], task["error"] = "failed", f"Planning failed: {err}"
        self._store.save(task)
        return task

    def start(self, objective: str, steps: list[str]) -> dict[str, Any]:
        """Run a plan the operator already approved in chat."""
        if self.active_task_id():
            raise RuntimeError("Another task is already driving the robot.")
        task = {
            "id": uuid.uuid4().hex[:12],
            "objective": objective.strip(),
            "status": "running",
            "steps": _steps(steps),
            "replans": 0,
            "error": "",
            "created": time.time(),
        }
        self._store.save(task)
        self._start(task["id"])
        return task

    def active_task_id(self) -> str | None:
        with self._lock:
            return self._active if self._worker is not None and self._worker.is_alive() else None

    # ── control ──────────────────────────────────────────────────────────────

    def get(self, task_id: str) -> dict[str, Any] | None:
        return self._store.get(task_id)

    def list(self) -> list[dict[str, Any]]:
        return self._store.list()

    def approve(self, task_id: str) -> dict[str, Any]:
        task = self._require(task_id)
        if task["status"] not in ("awaiting_approval", "paused"):
            raise ValueError(f"Task is {task['status']}, not waiting for approval.")
        task["status"] = "running"
        self._store.save(task)
        self._start(task_id)
        return task

    def cancel(self, task_id: str) -> dict[str, Any]:
        task = self._require(task_id)
        if task["status"] in ("done", "failed", "cancelled"):
            return task
        self._cancel.add(task_id)
        if self._current_thread and self._current_thread.startswith(task_id):
            cancel_run(self._current_thread)
        if self._worker is None or not self._worker.is_alive():
            task["status"] = "cancelled"
            self._store.save(task)
        return task

    def resume_interrupted(self) -> None:
        """After a restart, continue a task that was running (from its current step)."""
        for task in self._store.list():
            if task["status"] == "running":
                self._start(task["id"])
                return

    def _require(self, task_id: str) -> dict[str, Any]:
        task = self._store.get(task_id)
        if task is None:
            raise KeyError(task_id)
        return task

    def _start(self, task_id: str) -> None:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                raise RuntimeError("Another task is already driving the robot.")
            self._active = task_id
            self._worker = threading.Thread(target=self._run, args=(task_id,), daemon=True)
            self._worker.start()

    # ── execution ────────────────────────────────────────────────────────────

    def _run(self, task_id: str) -> None:
        task = self._require(task_id)
        try:
            while task["status"] == "running":
                if task_id in self._cancel:
                    task["status"] = "cancelled"
                    break
                index = next((i for i, s in enumerate(task["steps"]) if s["status"] != "done"), None)
                if index is None:
                    task["status"] = "done"
                    break
                self._run_step(task, index)
                self._store.save(task)
        except Exception as err:  # noqa: BLE001 - a crashed worker must leave a readable state
            _log.exception("task %s crashed", task_id)
            task["status"], task["error"] = "failed", f"{type(err).__name__}: {err}"
        finally:
            self._cancel.discard(task_id)
            self._store.save(task)

    def _run_step(self, task: dict[str, Any], index: int) -> None:
        step = task["steps"][index]
        step["status"] = "running"
        step["attempts"] += 1
        thread_id = f"{task['id']}-s{index + 1}-a{step['attempts']}"
        step["threads"].append(thread_id)
        self._store.save(task)
        done = [s["text"] for s in task["steps"] if s["status"] == "done"]
        prompt = _STEP.format(
            index=index + 1,
            total=len(task["steps"]),
            objective=task["objective"],
            done="; ".join(done) or "nothing yet",
            step=step["text"],
        )
        loop = self._make_loop()
        if _GRASP_STEP.search(step["text"]) and not _GRASP_TOOLS & set(loop.registry.by_name()):
            step["status"], step["evidence"] = "failed", "No tool can pick up or release objects yet."
            task["status"], task["error"] = "failed", f"Step '{step['text']}' needs a gripper tool, which does not exist yet."
            return
        self._current_thread = thread_id
        begin_run(thread_id)  # the chat Stop button (/chat/cancel) also stops the task
        try:
            result = loop.run([{"role": "user", "content": prompt}], thread_id=thread_id)
        except RunCancelled:
            task["status"], step["status"] = "cancelled", "failed"
            return
        finally:
            end_run(thread_id)
            self._current_thread = None

        verdict = self._verify(step["text"], result)
        step["evidence"] = verdict.evidence
        if verdict.ok:
            step["status"] = "done"
            return
        step["status"] = "failed"
        if step["attempts"] < MAX_ATTEMPTS:
            step["status"] = "pending"  # retry the same step
        elif task["replans"] < MAX_REPLANS:
            task["replans"] += 1
            new = self._plan(task["objective"], done, f"{step['text']}: {verdict.evidence}")
            task["steps"] = [s for s in task["steps"] if s["status"] == "done"] + _steps(new)
        else:
            task["status"] = "failed"
            task["error"] = f"Step '{step['text']}' failed: {verdict.evidence}"

    def _verify(self, step: str, result: Any) -> StepVerdict:
        if not result.tool_calls:
            return StepVerdict(ok=False, evidence="No tool was called for this step.")
        if all(c["name"] in _LOOKUPS for c in result.tool_calls):
            return StepVerdict(ok=False, evidence="Only looked things up; the step's action was not done.")
        calls = [
            {"tool": c["name"], "args": c["args"], "result": compact_tool_result(c["result"])}
            for c in result.tool_calls
        ]
        prompt = (
            f"Step: {step}\nTool calls: {json.dumps(calls, default=str)[:6000]}\n"
            f"Agent's final words: {result.response}"
        )
        try:
            return self._verifier.run_sync(prompt).output
        except Exception as err:  # noqa: BLE001 - an unverifiable step counts as failed
            return StepVerdict(ok=False, evidence=f"Could not verify: {err}")
