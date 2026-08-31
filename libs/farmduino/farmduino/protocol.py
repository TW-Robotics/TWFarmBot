"""Farmduino wire format: G/E/F commands out, R reports in.

See https://github.com/FarmBot/farmbot-arduino-firmware
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

_TOKEN = re.compile(r"([A-Za-z@]+)(-?\d+(?:\.\d+)?)?")
_PARAM = re.compile(r"([A-Z]{1,2})(-?\d+(?:\.\d+)?)")

# Command finished / failed / rejected. Listener stops waiting after these.
COMPLETION_CODES = frozenset({2, 3, 9, 20, 86})

PARAM_CONFIG_OK = 2


def queue_suffix_enabled() -> bool:
    return os.getenv("FARMBOT_QUEUE_SUFFIX", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


@dataclass(frozen=True)
class Command:
    kind: str
    number: int | None
    params: dict[str, float] = field(default_factory=dict)
    raw: str = ""

    @property
    def code(self) -> str:
        if self.kind == "E" or self.kind == "@":
            return self.kind
        return f"{self.kind}{self.number:02d}" if self.number is not None else self.kind


@dataclass(frozen=True)
class Report:
    number: int
    params: dict[str, float] = field(default_factory=dict)
    raw: str = ""
    message: str = ""

    @property
    def code(self) -> str:
        return f"R{self.number:02d}"

    @property
    def complete(self) -> bool:
        return self.number in COMPLETION_CODES or self.number == 87

    @property
    def ok(self) -> bool:
        return self.number in {2, 11, 12, 13, 20}


def encode(code: str, **params: Any) -> str:
    """Build a Farmduino line, e.g. encode('G00', X=10, Y=0, Z=0)."""
    parts = [code.strip().upper()]
    queue = params.pop("Q", params.pop("q", None))
    for key, value in params.items():
        if value is None:
            continue
        name = key.upper()
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        parts.append(f"{name}{value}")
    if queue is not None:
        parts.append(f"Q{int(queue)}")
    elif queue_suffix_enabled():
        parts.append("Q0")
    return " ".join(parts)


def parse_command(line: str) -> Command:
    text = line.strip()
    if not text:
        raise ValueError("empty command")
    head, *rest = text.split()
    kind = head[0].upper()
    if kind in {"E", "@"} and len(head) == 1:
        return Command(kind=kind, number=None, params=_params(" ".join(rest)), raw=text)
    if kind not in {"G", "F"}:
        raise ValueError(f"not a G/E/F command: {text}")
    number = int(head[1:] or "0")
    return Command(
        kind=kind,
        number=number,
        params=_params(" ".join(rest)),
        raw=text,
    )


def parse_report(line: str) -> Report | None:
    text = line.strip()
    if not text:
        return None
    if not text.upper().startswith("R"):
        return Report(number=-1, raw=text, message=text)
    match = _TOKEN.match(text)
    if match is None:
        return Report(number=-1, raw=text, message=text)
    number = int(match.group(2) or 0)
    rest = text[match.end() :].strip()
    return Report(number=number, params=_params(rest), raw=text, message=rest)


def _params(fragment: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, raw in _PARAM.findall(fragment.upper()):
        out[name] = float(raw)
    return out
