"""Tests for the restricted programmatic tool-calling runtime."""

from __future__ import annotations

from pydantic import BaseModel

from planning_service.harness.farm_script import (
    FarmScriptRuntime,
    extract_farm_scripts,
)
from planning_service.harness.tool_policy import ToolCategory, ToolDescriptor, ToolPolicy


class _MoveArgs(BaseModel):
    x: float
    y: float
    z: float = 0


class _NoArgs(BaseModel):
    pass


def _runtime() -> tuple[FarmScriptRuntime, list[tuple[str, dict]]]:
    log: list[tuple[str, dict]] = []

    def invoke(name: str, args: dict) -> dict:
        log.append((name, args))
        if name == "list_zones":
            return {
                "zones": [
                    {"name": "Tomato Zone", "cx": 10, "cy": 20},
                    {"name": "Herbs", "cx": 1, "cy": 2},
                ]
            }
        return {"status": "proposed", "kind": name, "params": args}

    descriptors = [
        ToolDescriptor(
            name="move",
            args_schema=_MoveArgs,
            policy=ToolPolicy(ToolCategory.ACT, requires_approval=True, description="Move"),
        ),
        ToolDescriptor(
            name="list_zones",
            args_schema=_NoArgs,
            policy=ToolPolicy(ToolCategory.READ, description="Zones"),
        ),
    ]
    return FarmScriptRuntime(descriptors, invoke), log


def test_extracts_python_fence() -> None:
    text = "Sure.\n```python\nmove(x=1, y=2, z=0)\n```\n"
    assert extract_farm_scripts(text) == ["move(x=1, y=2, z=0)"]


def test_extracts_bare_script_when_it_calls_a_tool() -> None:
    src = "move(x=1, y=2, z=0)"
    assert extract_farm_scripts(src, {"move"}) == [src]


def test_does_not_treat_json_as_script() -> None:
    src = '{"actions": [{"kind": "move"}]}'
    assert extract_farm_scripts(src, {"move"}) == []


def test_runs_loop_over_filtered_zones() -> None:
    runtime, log = _runtime()
    result = runtime.run(
        """
zones = list_zones()["zones"]
tomatoes = [z for z in zones if "tomato" in z["name"].lower()]
for z in tomatoes:
    move(x=z["cx"], y=z["cy"], z=0)
"""
    )
    assert result.ok, result.error
    assert [name for name, _ in log] == ["list_zones", "move"]
    assert log[1][1] == {"x": 10.0, "y": 20.0, "z": 0.0}


def test_rejects_import() -> None:
    runtime, _log = _runtime()
    result = runtime.run("import os\nmove(x=0, y=0, z=0)")
    assert result.ok is False
    assert "disallowed" in result.error.lower() or "import" in result.error.lower()


def test_rejects_dunder_and_open() -> None:
    runtime, log = _runtime()
    result = runtime.run("open('/etc/passwd')")
    assert result.ok is False
    assert log == []
    result = runtime.run("(1).__class__")
    assert result.ok is False


def test_last_expression_is_returned() -> None:
    runtime, _log = _runtime()
    result = runtime.run("1 + 2")
    assert result.ok
    assert result.result == 3
