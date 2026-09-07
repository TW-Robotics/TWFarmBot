"""Tests for the agentic long-term memory store and introspection tools."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from planning_service.harness import ContextBuilder, ToolRegistry
from planning_service.introspection import (
    InMemorySystemStateProvider,
    build_introspection_tools,
)
from planning_service.memory import MemoryStore
from twfarmbot_api_server.app import create_app
from twfarmbot_core.actions import ActionRegistry


@pytest.fixture
def mem_store() -> MemoryStore:
    return MemoryStore(db_path=":memory:")


def test_remember_and_get(mem_store: MemoryStore) -> None:
    res = mem_store.remember(
        content="Tomato plant at (1200, 450) exhibits mild yellowing.",
        category="agronomy",
        x=1200.0,
        y=450.0,
        z=10.0,
        tags=["tomato", "yellowing", "chlorosis"],
        importance=0.8,
    )
    assert res["status"] == "stored"
    assert res["id"] is not None
    assert res["category"] == "agronomy"
    assert res["importance"] == 0.8
    assert "yellowing" in res["tags"]

    fetched = mem_store.get(res["id"])
    assert fetched is not None
    assert fetched["content"] == "Tomato plant at (1200, 450) exhibits mild yellowing."
    assert fetched["x"] == 1200.0
    assert fetched["y"] == 450.0


def test_remember_empty_content_raises(mem_store: MemoryStore) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        mem_store.remember(content="   ")


def test_forget_memory(mem_store: MemoryStore) -> None:
    m = mem_store.remember(content="Temporary note", category="hardware")
    m_id = m["id"]
    assert mem_store.get(m_id) is not None

    ok = mem_store.forget(m_id)
    assert ok is True
    assert mem_store.get(m_id) is None
    # Subsequent forget on same id returns False
    assert mem_store.forget(m_id) is False


def test_spatial_recall_and_distance_ordering(mem_store: MemoryStore) -> None:
    # Memory exactly at (1000, 500)
    m1 = mem_store.remember(
        content="Center plant healthy",
        x=1000.0,
        y=500.0,
        importance=0.5,
    )
    # Memory 50mm away at (1030, 540) -> hypot(30, 40) = 50.0mm
    m2 = mem_store.remember(
        content="Nearby plant slightly dry",
        x=1030.0,
        y=540.0,
        importance=0.9,
    )
    # Memory far away at (2000, 1000)
    m3 = mem_store.remember(
        content="Distant corner bed",
        x=2000.0,
        y=1000.0,
        importance=0.9,
    )

    # Search near (1000, 500) with radius 100mm
    results = mem_store.recall(near_x=1000.0, near_y=500.0, radius_mm=100.0)
    assert len(results) == 2
    ids = [r["id"] for r in results]
    assert m1["id"] in ids
    assert m2["id"] in ids
    assert m3["id"] not in ids

    # Sorted by closest distance first
    assert results[0]["id"] == m1["id"]
    assert results[0]["distance_mm"] == 0.0
    assert results[1]["id"] == m2["id"]
    assert results[1]["distance_mm"] == 50.0


def test_keyword_and_category_search(mem_store: MemoryStore) -> None:
    mem_store.remember(
        content="Watered zone 1 for 25 seconds after NDRE dropped to 0.12",
        category="treatment",
        tags=["water", "ndre", "low"],
        importance=0.7,
    )
    mem_store.remember(
        content="Soil moisture pin 14 measured 340 ADC",
        category="hardware",
        tags=["sensor", "moisture"],
        importance=0.3,
    )
    mem_store.remember(
        content="Do not water corner (100, 100) due to poor drainage",
        category="rule",
        tags=["drainage", "water_limit"],
        importance=0.9,
    )

    # Search keyword "NDRE"
    ndre_res = mem_store.recall(query="ndre")
    assert len(ndre_res) == 1
    assert "NDRE" in ndre_res[0]["content"]

    # Search keyword "water" across categories
    water_res = mem_store.recall(query="water")
    assert len(water_res) == 2

    # Filter by category
    rule_res = mem_store.recall(category="rule")
    assert len(rule_res) == 1
    assert "drainage" in rule_res[0]["content"]


def test_inmemory_provider_memory_tools() -> None:
    mem_store = MemoryStore(db_path=":memory:")
    provider = InMemorySystemStateProvider(memory_store=mem_store)
    tools = {t.name: t for t in build_introspection_tools(provider)}

    assert "remember" in tools
    assert "recall_memories" in tools
    assert "forget_memory" in tools

    # 1. Call remember tool
    remember_tool = tools["remember"]
    res = remember_tool.invoke(
        {
            "content": "Observed rapid leaf recovery after watering.",
            "category": "agronomy",
            "x": 800.0,
            "y": 400.0,
            "tags": ["recovery"],
            "importance": 0.8,
        }
    )
    assert res["status"] == "stored"
    mem_id = res["id"]

    # 2. Call recall_memories tool
    recall_tool = tools["recall_memories"]
    recalled = recall_tool.invoke({"near_x": 800.0, "near_y": 400.0, "radius_mm": 50.0})
    assert recalled["count"] == 1
    assert recalled["memories"][0]["id"] == mem_id

    # 3. Call forget_memory tool
    forget_tool = tools["forget_memory"]
    forgot = forget_tool.invoke({"memory_id": mem_id})
    assert forgot["forgotten"] is True

    # Confirm it is no longer recalled
    after_forget = recall_tool.invoke({"near_x": 800.0, "near_y": 400.0})
    assert after_forget["count"] == 0


def test_context_builder_injects_memories(mem_store: MemoryStore) -> None:
    mem_store.remember(
        content="Plant at (1100, 200) showed high chlorophyll index.",
        category="agronomy",
        x=1100.0,
        y=200.0,
    )
    reg = ActionRegistry()
    provider = InMemorySystemStateProvider(memory_store=mem_store)
    tool_reg = ToolRegistry(reg, system_state=provider)
    cb = ContextBuilder(tool_reg)

    prompt = cb.chat_system_prompt()
    assert "Recent Long-Term Memories" in prompt
    assert "chlorophyll index" in prompt
    assert "ACTIVELY REASON" in prompt


def test_collapse_auto_entries_keeps_one_water_and_one_ndre_cell() -> None:
    store = MemoryStore(db_path=":memory:")
    store.remember("manual note", tags=["operator"])
    store.remember("2026-09-07 11:00Z Pump ran for the whole bed (~60s) at x=0, y=0.", tags=["auto", "water"])
    store.remember("2026-09-07 11:56Z Pump ran for the whole bed (~60s) at x=250, y=0.", tags=["auto", "water"])
    store.remember(
        "2026-09-07 11:45Z consider_water, mean=-0.02 at x=0, y=0.",
        x=0,
        y=0,
        tags=["auto", "ndre"],
    )
    store.remember(
        "2026-09-07 11:46Z consider_water, mean=-0.02 at x=50, y=0.",
        x=50,
        y=0,
        tags=["auto", "ndre"],
    )
    store.remember(
        "2026-09-07 11:50Z consider_water, mean=-0.01 at x=500, y=0.",
        x=500,
        y=0,
        tags=["auto", "ndre"],
    )
    forgotten = store.collapse_auto_entries()
    assert forgotten >= 2
    leftover = store.list_recent(limit=20)
    assert any("manual note" in item["content"] for item in leftover)
    waters = [item for item in leftover if "water" in item["tags"] and "auto" in item["tags"]]
    assert len(waters) == 1
    ndres = [item for item in leftover if "ndre" in item["tags"]]
    assert len(ndres) == 2


def test_api_server_memory_endpoints(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    db_file = tmp_path / "test_api_memory.db"
    monkeypatch.setenv("FARMBOT_MEMORY_DB", str(db_file))

    # Reset global singleton to point to new test db
    import planning_service.memory as psm

    psm._GLOBAL_STORE = None

    reg = ActionRegistry()
    app = create_app(registry=reg)
    client = TestClient(app)

    # 1. POST /memory/remember
    r = client.post(
        "/memory/remember",
        json={
            "content": "Operator instruction: maintain high soil moisture for lettuce bed.",
            "category": "rule",
            "x": 300.0,
            "y": 600.0,
            "tags": ["lettuce", "moisture"],
            "importance": 0.9,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "stored"
    mem_id = data["id"]

    # 2. GET /memory
    r_get = client.get("/memory", params={"query": "lettuce"})
    assert r_get.status_code == 200
    get_data = r_get.json()
    assert get_data["count"] == 1
    assert get_data["memories"][0]["id"] == mem_id

    # 3. Spatial GET /memory
    r_spatial = client.get("/memory", params={"near_x": 300.0, "near_y": 600.0, "radius_mm": 50.0})
    assert r_spatial.status_code == 200
    assert r_spatial.json()["count"] == 1

    # 4. DELETE /memory/{id}
    r_del = client.delete(f"/memory/{mem_id}")
    assert r_del.status_code == 200
    assert r_del.json()["forgotten"] is True

    # 5. Confirm deleted
    r_check = client.get("/memory", params={"query": "lettuce"})
    assert r_check.json()["count"] == 0

    psm._GLOBAL_STORE = None
