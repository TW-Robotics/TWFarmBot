"""Generic end effector call: config-driven commands on a sim or Modbus backend."""

from __future__ import annotations

import pytest

from watering_service import end_effectors

_GRIPPER = {
    "backend": "sim",
    "devices": {
        "gripper": {
            "commands": {"grip": [{"coil": 0, "value": 1}], "release": [{"coil": 0, "value": 0}]},
            "status": {"holding": {"discrete_input": 0}},
        }
    },
}


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setattr(end_effectors, "load_yaml_config", lambda: {"end_effectors": _GRIPPER})
    monkeypatch.setattr(end_effectors, "_backend", None)


def test_command_runs_and_reports_status(configured) -> None:
    assert end_effectors.run("gripper", "grip")["status"] == {"holding": 1}
    assert end_effectors.run("gripper", "release")["status"] == {"holding": 0}
    assert end_effectors.describe() == {"gripper": {"commands": ["grip", "release"], "status": ["holding"]}}


def test_unknown_device_or_command_is_refused(configured) -> None:
    with pytest.raises(ValueError, match="No end effector"):
        end_effectors.run("vacuum", "on")
    with pytest.raises(ValueError, match="no command"):
        end_effectors.run("gripper", "explode")


class _Response:
    bits = [True]
    registers = [7]

    def isError(self) -> bool:
        return False


class _FakeClient:
    """Stands in for pymodbus' ModbusTcpClient; records calls for assertions."""

    def __init__(self, calls: list[tuple], host, port, timeout) -> None:
        self.calls = calls
        calls.append(("init", host, port))

    def connect(self) -> bool:
        return True

    def write_coil(self, address, value):
        self.calls.append(("write_coil", address, value))
        return _Response()

    def write_register(self, address, value):
        self.calls.append(("write_register", address, value))
        return _Response()

    def read_discrete_inputs(self, address, count):
        self.calls.append(("read_discrete_inputs", address))
        return _Response()

    def read_input_registers(self, address, count):
        self.calls.append(("read_input_registers", address))
        return _Response()

    def read_holding_registers(self, address, count):
        self.calls.append(("read_holding_registers", address))
        return _Response()


def test_modbus_backend_maps_writes_and_reads(configured, monkeypatch) -> None:
    calls: list[tuple] = []
    monkeypatch.setattr("pymodbus.client.ModbusTcpClient", lambda h, port, timeout: _FakeClient(calls, h, port, timeout))
    cfg = {**_GRIPPER, "backend": "modbus", "modbus": {"host": "10.0.0.5"}}
    monkeypatch.setattr(end_effectors, "load_yaml_config", lambda: {"end_effectors": cfg})
    assert end_effectors.run("gripper", "grip")["status"] == {"holding": 1}
    assert calls == [("init", "10.0.0.5", 502), ("write_coil", 0, True), ("read_discrete_inputs", 0)]


def test_modbus_backend_register_commands(monkeypatch) -> None:
    """The RevPi ModbusTCP Slave exposes words, not coils: cover that path."""
    calls: list[tuple] = []
    monkeypatch.setattr("pymodbus.client.ModbusTcpClient", lambda h, port, timeout: _FakeClient(calls, h, port, timeout))
    cfg = {
        "backend": "modbus",
        "modbus": {"host": "10.0.0.5"},
        "devices": {
            "tool": {
                "commands": {"on": [{"register": 0, "value": 1}], "off": [{"register": 0, "value": 0}]},
                "status": {"active": {"register": 0}},
            }
        },
    }
    monkeypatch.setattr(end_effectors, "load_yaml_config", lambda: {"end_effectors": cfg})
    monkeypatch.setattr(end_effectors, "_backend", None)
    assert end_effectors.run("tool", "on")["status"] == {"active": 7}
    assert calls == [("init", "10.0.0.5", 502), ("write_register", 0, 1), ("read_holding_registers", 0)]


def test_dev_yaml_command_names_survive_yaml_parsing() -> None:
    """Unquoted on:/off: in YAML parse as booleans; the config must quote them."""
    from twfarmbot_core.config import load_yaml_config

    devices = load_yaml_config().get("end_effectors", {}).get("devices", {})
    assert devices, "expected at least one end effector in configs/dev.yaml"
    for name, device in devices.items():
        for command in device.get("commands", {}):
            assert isinstance(command, str), f"{name!r} has non-string command {command!r}"


def test_configured_device_registers_agent_tool() -> None:
    from twfarmbot_api_server.handlers import register_default_handlers
    from twfarmbot_core.actions import ActionRegistry

    registry = ActionRegistry()
    register_default_handlers(registry)
    assert "end_effector" in registry.kinds()  # configs/dev.yaml ships a tool device
