"""End effectors on the RevPi (or simulated), driven by one generic call.

Every device and command is described in ``end_effectors`` in the YAML config,
so a gripper, vacuum or cutter is a config block, not new code::

    end_effectors:
      backend: modbus            # modbus (RevPi Modbus TCP server) | sim
      modbus: {host: 192.168.1.50, port: 502, timeout_s: 2}
      devices:
        gripper:
          commands:
            grip:    [{coil: 0, value: 1}]
            release: [{coil: 0, value: 0}]
          status:
            holding: {discrete_input: 0}

A command is a list of writes (``coil`` or ``register``). ``status`` maps a
name to one read (``coil``, ``discrete_input``, ``register``,
``input_register``); it is returned after every command, so a task step can
be checked against real feedback.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from twfarmbot_core.config import load_yaml_config, yaml_config_path

log = logging.getLogger(__name__)

_WRITES = ("coil", "register")
_READS = ("coil", "discrete_input", "register", "input_register")


def _cfg() -> dict[str, Any]:
    cfg = load_yaml_config().get("end_effectors") or {}
    return cfg if isinstance(cfg, dict) else {}


def devices() -> dict[str, dict[str, Any]]:
    return dict(_cfg().get("devices") or {})


def describe() -> dict[str, dict[str, list[str]]]:
    """Device -> its commands and status fields (for tool descriptions and the UI)."""
    return {
        name: {"commands": sorted(dev.get("commands") or {}), "status": sorted(dev.get("status") or {})}
        for name, dev in devices().items()
    }


def rename(name: str, label: str) -> None:
    """Set a device's UI label in the YAML config, keeping comments and layout."""
    from ruamel.yaml import YAML

    label = label.strip()
    if not label:
        raise ValueError("label must not be empty")
    path = yaml_config_path()
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096
    with path.open(encoding="utf-8") as fh:
        data = yaml.load(fh)
    device = ((data.get("end_effectors") or {}).get("devices") or {}).get(name)
    if device is None:
        raise ValueError(f"No end effector named {name!r}. Configured: {sorted(devices())}")
    device["label"] = label
    with path.open("w", encoding="utf-8") as fh:
        yaml.dump(data, fh)


class SimBackend:
    """No hardware: remembers writes; a read returns what was written to that address."""

    def __init__(self) -> None:
        self.memory: dict[int, int] = {}

    def write(self, kind: str, address: int, value: int) -> None:
        self.memory[address] = value

    def read(self, kind: str, address: int) -> int:
        return self.memory.get(address, 0)


class ModbusBackend:
    """RevPi built-in Modbus TCP server (configured in PiCtory)."""

    def __init__(self, host: str, port: int = 502, timeout_s: float = 2.0) -> None:
        from pymodbus.client import ModbusTcpClient

        self.client = ModbusTcpClient(host, port=port, timeout=timeout_s)

    def _ok(self, response: Any) -> Any:
        if response is None or response.isError():
            raise ConnectionError(f"RevPi Modbus error: {response}")
        return response

    def write(self, kind: str, address: int, value: int) -> None:
        if not self.client.connect():
            raise ConnectionError("RevPi unreachable")
        if kind == "coil":
            self._ok(self.client.write_coil(address, bool(value)))
        else:
            self._ok(self.client.write_register(address, int(value)))

    def read(self, kind: str, address: int) -> int:
        if not self.client.connect():
            raise ConnectionError("RevPi unreachable")
        method = {
            "coil": "read_coils",
            "discrete_input": "read_discrete_inputs",
            "register": "read_holding_registers",
            "input_register": "read_input_registers",
        }[kind]
        response = self._ok(getattr(self.client, method)(address, count=1))
        values = getattr(response, "bits", None) if kind in ("coil", "discrete_input") else response.registers
        return int(values[0])


_backend: Any | None = None


def get_backend() -> Any:
    global _backend
    if _backend is None:
        cfg = _cfg()
        name = (os.getenv("END_EFFECTOR_BACKEND") or str(cfg.get("backend") or "sim")).lower()
        if name == "modbus":
            modbus = cfg.get("modbus") or {}
            host = os.getenv("REVPI_HOST") or str(modbus.get("host") or "")
            if not host:
                raise ValueError("Modbus backend needs end_effectors.modbus.host or REVPI_HOST")
            _backend = ModbusBackend(host, int(modbus.get("port", 502)), float(modbus.get("timeout_s", 2.0)))
        else:
            _backend = SimBackend()
        log.info("end effector backend: %s", type(_backend).__name__)
    return _backend


def _io(spec: dict[str, Any], kinds: tuple[str, ...]) -> tuple[str, int]:
    kind = next((k for k in kinds if k in spec), None)
    if kind is None:
        raise ValueError(f"I/O spec {spec} needs one of {kinds}")
    return kind, int(spec[kind])


def status(name: str) -> dict[str, int]:
    device = devices().get(name)
    if device is None:
        raise ValueError(f"No end effector named {name!r}. Configured: {sorted(devices())}")
    backend = get_backend()
    return {field: backend.read(*_io(spec, _READS)) for field, spec in (device.get("status") or {}).items()}


def run(name: str, command: str) -> dict[str, Any]:
    """Run one configured command, then read the device's status back."""
    device = devices().get(name)
    if device is None:
        raise ValueError(f"No end effector named {name!r}. Configured: {sorted(devices())}")
    steps = (device.get("commands") or {}).get(command)
    if steps is None:
        raise ValueError(f"{name!r} has no command {command!r}. Commands: {sorted(device.get('commands') or {})}")
    backend = get_backend()
    for step in steps:
        kind, address = _io(step, _WRITES)
        backend.write(kind, address, int(step.get("value", 1)))
    log.info("end effector %s: %s", name, command)
    return {"device": name, "command": command, "status": status(name)}
