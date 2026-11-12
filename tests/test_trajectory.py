"""Trajectory log: append-only JSONL per agent run, readable back."""

from __future__ import annotations

import json
from pathlib import Path

from planning_service.harness import (
    AgentLoop,
    ApprovalGate,
    ContextBuilder,
    ToolRegistry,
)
from planning_service.harness.trajectory import (
    TrajectoryLog,
    list_trajectories,
    read_trajectory,
)
from tests.fakes import ScriptModel
from tests.test_harness import _make_registry, _tool_call
from twfarmbot_core.actions import ActionRegistry


def test_trajectory_log_appends_and_reads_back(tmp_path: Path) -> None:
    log = TrajectoryLog("thread-1", tmp_path, model_name="fake-model")
    log.record("run_start", prompt="take a photo")
    log.record("tool_call", name="take_photo", args={}, result={"status": "ok"})
    log.record("run_end", stop_reason="done", response="photo taken")

    records = read_trajectory("thread-1", tmp_path)
    assert [r["type"] for r in records] == ["run_start", "tool_call", "run_end"]
    assert [r["seq"] for r in records] == [0, 1, 2]
    assert all(r["thread_id"] == "thread-1" and r["ts"] for r in records)
    assert records[0]["model"] == "fake-model"

    listed = list_trajectories(tmp_path)
    assert listed[0]["thread_id"] == "thread-1"
    assert listed[0]["prompt"] == "take a photo"
    assert listed[0]["stop_reason"] == "done"
    assert listed[0]["records"] == 3


def test_read_trajectory_rejects_path_traversal(tmp_path: Path) -> None:
    assert read_trajectory("../secret", tmp_path) == []


def test_trajectory_endpoints(tmp_path: Path, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from twfarmbot_api_server.app import create_app

    monkeypatch.setenv("TWFB_TRAJECTORY_DIR", str(tmp_path))
    log = TrajectoryLog("api-thread", tmp_path, model_name="fake-model")
    log.record("run_start", prompt="water the carrots")
    log.record("run_end", stop_reason="done", metrics={"total_latency_s": 1.5})

    client = TestClient(create_app())
    listed = client.get("/trajectories")
    assert listed.status_code == 200
    assert listed.json()["trajectories"][0]["thread_id"] == "api-thread"

    detail = client.get("/trajectories/api-thread")
    assert detail.status_code == 200
    assert detail.json()["count"] == 2
    assert client.get("/trajectories/missing").status_code == 404


def _make_loop(model: ScriptModel, reg: ActionRegistry, tmp_path: Path) -> AgentLoop:
    tool_registry = ToolRegistry(reg)
    return AgentLoop(
        model=model,
        core_tools=frozenset(tool_registry.by_name()),
        tool_registry=tool_registry,
        approval_gate=ApprovalGate(reg),
        context_builder=ContextBuilder(tool_registry),
        propose_only=True,
        allow_actions=True,
        model_name="fake-model",
        trajectory_dir=tmp_path,
    )


def test_agent_loop_stream_writes_trajectory(tmp_path: Path) -> None:
    fake = ScriptModel(responses=["unused"])
    fake.set_responses([_tool_call("take_photo", {}, "photo_1"), "photo taken"])
    loop = _make_loop(fake, _make_registry(), tmp_path)
    events = list(
        loop.stream([{"role": "user", "content": "take a photo"}], thread_id="t-stream")
    )
    assert any(e["type"] == "meta" for e in events)

    records = read_trajectory("t-stream", tmp_path)
    types = [r["type"] for r in records]
    assert types[0] == "run_start"
    assert "tool_start" in types and "tool_call" in types
    assert types[-1] == "run_end"
    end = records[-1]
    assert end["stop_reason"] == "done"
    assert end["metrics"]["turn_count"] >= 1
    tool = next(r for r in records if r["type"] == "tool_call")
    assert tool["name"] == "take_photo"
    assert tool["result"]["status"] == "ok"
    assert tool["duration_s"] >= 0


def test_agent_loop_run_writes_trajectory(tmp_path: Path) -> None:
    fake = ScriptModel(responses=["unused"])
    fake.set_responses([_tool_call("take_photo"), "done"])
    loop = _make_loop(fake, _make_registry(), tmp_path)
    result = loop.run([{"role": "user", "content": "take a photo"}])
    assert result.response == "done"

    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    records = [json.loads(line) for line in files[0].read_text().splitlines()]
    assert records[0]["type"] == "run_start"
    assert records[-1]["type"] == "run_end"
    assert records[-1]["response"] == "done"
