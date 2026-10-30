"""Tests for closed-loop See → Reason → Act → Reason autonomous cycle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from planning_service import AutonomousCycleResult, run_autonomous_cycle
from safety_service import validate as safety_validate
from tests.fakes import AIMessage, ScriptModel
from twfarmbot_api_server.app import create_app
from twfarmbot_core.actions import ActionRegistry
from twfarmbot_core.domain import Action
from twfarmbot_worker.inspect_job import run_autonomous_loop


def _make_test_registry() -> tuple[ActionRegistry, list[Action]]:
    dispatched: list[Action] = []

    def fake_capture_ndre(action: Action) -> Action:
        dispatched.append(action)
        return Action(
            kind="capture_ndre",
            params={
                **action.params,
                "ndre": {
                    "mean": 0.08,
                    "vegetation_fraction": 0.35,
                    "stress_fraction": 0.55,
                },
                "interpretation": {
                    "label": "likely_stressed",
                    "action_hint": "consider_water",
                    "advice": "High low-vigor fraction — canopy may be water-stressed.",
                },
            },
        )

    def fake_water(action: Action) -> Action:
        dispatched.append(action)
        return action

    registry = ActionRegistry(validate=safety_validate)
    registry.register("capture_ndre", fake_capture_ndre)
    registry.register("water", fake_water)
    return registry, dispatched


def test_autonomous_cycle_sees_reasons_acts_reasons_when_stressed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When canopy is stressed, the agent sees NDRE, reasons, waters, and verifies."""
    monkeypatch.setenv("TWFB_UI_DATA_DIR", str(tmp_path))

    registry, dispatched = _make_test_registry()

    fake_model = ScriptModel(
        turns=[
            # 1. SEE
            AIMessage(
                content="Inspecting canopy health first.",
                tool_calls=[{"name": "capture_ndre", "args": {"return_to_start": True}, "id": "tc_1"}],
            ),
            # 2. REASON & ACT
            AIMessage(
                content="NDRE shows water stress (stress_fraction=0.55). Watering bed.",
                tool_calls=[{"name": "water", "args": {"seconds": 25.0}, "id": "tc_2"}],
            ),
            # 3. REASON & VERIFY
            AIMessage(
                content="Completed autonomous crop care: detected low vigor and delivered 25s water. Canopy status updated.",
            ),
        ]
    )

    result: AutonomousCycleResult = run_autonomous_cycle(
        registry=registry,
        model=fake_model,
        save_session=True,
    )

    assert result.status == "ok"
    # Perceptions: SEE
    assert len(result.perceptions) == 1
    assert result.perceptions[0]["name"] == "capture_ndre"
    # Actions: ACT
    assert len(result.actions) == 1
    assert result.actions[0]["name"] == "water"
    assert result.actions[0]["args"]["seconds"] == 25.0
    # Dispatched physically
    assert [a.kind for a in dispatched] == ["capture_ndre", "water"]
    # Final synthesis
    assert "delivered 25s water" in result.summary
    # Session file persisted
    assert result.session_path is not None
    session_file = Path(result.session_path)
    assert session_file.is_file()
    data = json.loads(session_file.read_text())
    assert data["kind"] == "autonomous"
    assert len(data["autonomous_cycle"]["actions"]) == 1


def test_autonomous_cycle_sees_reasons_does_not_water_when_healthy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When canopy is healthy, the agent sees NDRE, reasons, and avoids unnecessary watering."""
    monkeypatch.setenv("TWFB_UI_DATA_DIR", str(tmp_path))

    dispatched: list[Action] = []

    def fake_capture_ndre(action: Action) -> Action:
        dispatched.append(action)
        return Action(
            kind="capture_ndre",
            params={
                **action.params,
                "ndre": {
                    "mean": 0.35,
                    "vegetation_fraction": 0.65,
                    "stress_fraction": 0.05,
                },
                "interpretation": {
                    "label": "healthy_canopy",
                    "action_hint": "ok",
                    "advice": "Healthy canopy signal — no watering needed.",
                },
            },
        )

    registry = ActionRegistry(validate=safety_validate)
    registry.register("capture_ndre", fake_capture_ndre)
    registry.register("water", lambda a: dispatched.append(a) or a)

    fake_model = ScriptModel(
        turns=[
            # 1. SEE
            AIMessage(
                content="Inspecting canopy health.",
                tool_calls=[{"name": "capture_ndre", "args": {"return_to_start": True}, "id": "tc_1"}],
            ),
            # 2. REASON (No ACT needed!)
            AIMessage(
                content="Canopy is healthy (NDRE 0.35, veg fraction 0.65). No watering required; monitoring continues.",
            ),
        ]
    )

    result: AutonomousCycleResult = run_autonomous_cycle(
        registry=registry,
        model=fake_model,
        save_session=True,
    )

    assert result.status == "ok"
    assert len(result.perceptions) == 1
    assert result.perceptions[0]["name"] == "capture_ndre"
    assert len(result.actions) == 0  # No water dispatched!
    assert [a.kind for a in dispatched] == ["capture_ndre"]
    assert "No watering required" in result.summary


def test_autonomous_cycle_safety_rejection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If the LLM attempts an unsafe action, the safety service gates it and returns feedback."""
    monkeypatch.setenv("TWFB_UI_DATA_DIR", str(tmp_path))

    registry, _ = _make_test_registry()

    fake_model = ScriptModel(
        turns=[
            # 1. SEE
            AIMessage(
                content="Checking health.",
                tool_calls=[{"name": "capture_ndre", "args": {"return_to_start": True}, "id": "tc_1"}],
            ),
            # 2. ATTEMPT UNSAFE ACT (water 9999s > 300s limit)
            AIMessage(
                content="Flooding bed.",
                tool_calls=[{"name": "water", "args": {"seconds": 9999.0}, "id": "tc_2"}],
            ),
            # 3. REASON on error feedback
            AIMessage(
                content="Watering exceeded safety limit and was blocked by safety service.",
            ),
        ]
    )

    result = run_autonomous_cycle(
        registry=registry,
        model=fake_model,
        save_session=False,
    )

    assert result.status == "ok"
    assert len(result.actions) == 1
    water_call = result.actions[0]
    assert water_call["result"]["status"] == "error"
    assert "unsafe" in water_call["result"]["error"].lower()


def test_autonomous_api_endpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /autonomous/run and GET /autonomous/status work via API."""
    monkeypatch.setenv("TWFB_UI_DATA_DIR", str(tmp_path))

    registry, _ = _make_test_registry()
    app = create_app(registry=registry)
    client = TestClient(app)

    fake_model = ScriptModel(
        turns=[
            AIMessage(content="Bed is in optimal condition."),
        ]
    )

    # Monkeypatch planner_chat to use fake model
    from planning_service import chat as real_chat

    def stub_chat(*args: Any, **kwargs: Any):
        kwargs["model"] = fake_model
        return real_chat(*args, **kwargs)

    monkeypatch.setattr("planning_service.autonomy.chat", stub_chat)

    res = client.post("/autonomous/run", json={"goal": "Routine check"})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert "optimal" in body["summary"]

    # Status check
    status_res = client.get("/autonomous/status")
    assert status_res.status_code == 200
    status_body = status_res.json()
    assert status_body["status"] == "ok"
    assert "latest_session" in status_body


def test_worker_run_autonomous_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Worker run_autonomous_loop sends POST /autonomous/run to API."""
    called = []

    class FakeHttpClient:
        def get(self, url: str):
            class Resp:
                def json(self):
                    return {"farmbot": "connected"}
            return Resp()

        def post(self, url: str, json: dict):
            called.append((url, json))
            class Resp:
                def raise_for_status(self):
                    pass
                def json(self):
                    return {"status": "ok", "actions": [{"kind": "water"}], "summary": "done"}
            return Resp()

    out = run_autonomous_loop(client=FakeHttpClient(), goal="Run morning care")
    assert out["status"] == "ok"
    assert len(called) == 1
    assert called[0][0] == "/autonomous/run"
    assert called[0][1]["goal"] == "Run morning care"


def test_autonomous_cycle_with_memory_recall_and_remember(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Agent in autonomous cycle recalls memory, acts, and commits new memory."""
    from planning_service.introspection import InMemorySystemStateProvider
    from planning_service.memory import MemoryStore

    monkeypatch.setenv("TWFB_UI_DATA_DIR", str(tmp_path))
    mem_store = MemoryStore(db_path=":memory:")
    # Seed prior memory
    mem_store.remember(
        content="Historical note: area previously suffered from nitrogen deficiency",
        category="agronomy",
        x=1200.0,
        y=450.0,
        importance=0.6,
    )
    provider = InMemorySystemStateProvider(memory_store=mem_store)
    registry, _ = _make_test_registry()

    fake_model = ScriptModel(
        answer_think=True,  # this cycle has a `think` tool, so the first step is forced
        turns=[
            # Step 1: Recall
            AIMessage(
                content="Checking historical records for this location...",
                tool_calls=[
                    {
                        "name": "recall_memories",
                        "args": {"near_x": 1200.0, "near_y": 450.0, "radius_mm": 100.0},
                        "id": "call_recall",
                    }
                ],
            ),
            # Step 2: See
            AIMessage(
                content="Now capturing current spectral vigor...",
                tool_calls=[
                    {"name": "capture_ndre", "args": {}, "id": "call_ndre"}
                ],
            ),
            # Step 3: Act
            AIMessage(
                content="High stress fraction (0.55). Delivering 20s water.",
                tool_calls=[
                    {"name": "water", "args": {"seconds": 20.0}, "id": "call_water"}
                ],
            ),
            # Step 4: Reason & Remember
            AIMessage(
                content="Commit outcome to long-term memory for multi-day monitoring.",
                tool_calls=[
                    {
                        "name": "remember",
                        "args": {
                            "content": "Watered 20s at (1200, 450) due to 0.55 NDRE stress.",
                            "category": "treatment",
                            "x": 1200.0,
                            "y": 450.0,
                            "tags": ["stress", "water"],
                            "importance": 0.8,
                        },
                        "id": "call_remember",
                    }
                ],
            ),
            # Step 5: Final Summary
            AIMessage(
                content="Completed autonomous cycle: recalled past history, detected stress, watered 20s, and saved treatment to long-term memory."
            ),
        ]
    )

    result = run_autonomous_cycle(
        registry=registry,
        model=fake_model,
        system_state=provider,
        save_session=True,
        ensure_homed=False,
    )

    assert result.status == "ok"
    seen = [p for p in result.perceptions if p["name"] != "think"]
    assert len(seen) == 1
    assert seen[0]["name"] == "capture_ndre"
    assert len(result.actions) == 1
    assert result.actions[0]["name"] == "water"
    assert len(result.memories) == 2
    memory_call_names = [m["name"] for m in result.memories]
    assert "recall_memories" in memory_call_names
    assert "remember" in memory_call_names

    # Verify memory was actually saved into the store
    saved = mem_store.recall(near_x=1200.0, near_y=450.0, category="treatment")
    assert len(saved) == 1
    assert "Watered 20s" in saved[0]["content"]
