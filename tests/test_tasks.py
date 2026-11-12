"""Long-horizon task runner: plan, approve once, run steps, retry, re-plan, resume."""

from __future__ import annotations

import json
from types import SimpleNamespace

from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from planning_service.tasks import TaskRunner, TaskStore


def _model(plans: list[list[str]], verdicts: list[bool]):
    """Planner answers come from ``plans``, verifier answers from ``verdicts``."""

    def respond(messages, info):
        tool = info.output_tools[0]
        if "steps" in json.dumps(tool.parameters_json_schema):
            args = {"steps": plans.pop(0)}
        else:
            ok = verdicts.pop(0)
            args = {"ok": ok, "evidence": "looked fine" if ok else "error result"}
        return ModelResponse(parts=[ToolCallPart(tool.name, args)])

    return FunctionModel(respond)


class _Loop:
    registry = SimpleNamespace(by_name=lambda: {"goto_named": None})

    def __init__(self, log: list[str]) -> None:
        self.log = log

    def run(self, messages, *, thread_id):
        self.log.append(thread_id)
        call = {"name": "goto_named", "args": {"name": "Carrot"}, "result": {"status": "ok"}}
        return SimpleNamespace(tool_calls=[call], response="Moved.")


def _runner(tmp_path, plans, verdicts, log):
    return TaskRunner(lambda: _Loop(log), _model(plans, verdicts), TaskStore(tmp_path))


def _wait(runner: TaskRunner) -> None:
    runner._worker.join(timeout=10)


def test_task_waits_for_approval_then_runs_every_step(tmp_path) -> None:
    log: list[str] = []
    runner = _runner(tmp_path, [["Go to Carrot", "Take a photo"]], [True, True], log)
    task = runner.create("move the carrot")
    assert task["status"] == "awaiting_approval" and not log

    runner.approve(task["id"])
    _wait(runner)
    task = runner.get(task["id"])
    assert task["status"] == "done"
    assert [s["status"] for s in task["steps"]] == ["done", "done"]
    assert log == [f"{task['id']}-s1-a1", f"{task['id']}-s2-a1"]


def test_failed_step_is_retried_then_replanned(tmp_path) -> None:
    log: list[str] = []
    plans = [["Water the carrot"], ["Mount watering nozzle", "Water the carrot"]]
    runner = _runner(tmp_path, plans, [False, False, True, True], log)
    task = runner.create("get the carrot")
    runner.approve(task["id"])
    _wait(runner)
    task = runner.get(task["id"])
    assert task["status"] == "done" and task["replans"] == 1
    assert [s["text"] for s in task["steps"]] == ["Mount watering nozzle", "Water the carrot"]


def test_running_task_resumes_at_its_current_step(tmp_path) -> None:
    log: list[str] = []
    runner = _runner(tmp_path, [["A", "B"]], [True], log)
    task = runner.create("two steps")
    task["status"] = "running"
    task["steps"][0]["status"] = "done"
    TaskStore(tmp_path).save(task)

    runner.resume_interrupted()
    _wait(runner)
    assert log == [f"{task['id']}-s2-a1"]
    assert runner.get(task["id"])["status"] == "done"


def test_start_runs_an_already_approved_plan(tmp_path) -> None:
    log: list[str] = []
    runner = _runner(tmp_path, [], [True, True], log)
    task = runner.start("carrot photo", ["Go to Carrot", "Take a photo"])
    assert task["status"] == "running"
    _wait(runner)
    assert runner.get(task["id"])["status"] == "done" and len(log) == 2


def test_grasp_step_fails_up_front_without_a_gripper_tool(tmp_path) -> None:
    log: list[str] = []
    runner = _runner(tmp_path, [], [True], log)
    task = runner.start("carrot", ["Go to Carrot", "Let go of the carrot"])
    _wait(runner)
    task = runner.get(task["id"])
    assert task["status"] == "failed" and "gripper" in task["error"]
    assert [s["status"] for s in task["steps"]] == ["done", "failed"]
    assert len(log) == 1  # the grasp step never ran


def test_step_that_only_looks_things_up_is_not_done(tmp_path) -> None:
    runner = _runner(tmp_path, [], [], [])
    lookups = SimpleNamespace(tool_calls=[{"name": "get_status", "args": {}, "result": {}}], response="Nothing to do.")
    assert runner._verify("Water the carrot", lookups).ok is False
