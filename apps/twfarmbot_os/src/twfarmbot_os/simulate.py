"""In-memory Farmduino: same G/E/F in, R codes out."""

from __future__ import annotations

from farmduino import Command, encode, parse_command

_HOME = {"F11": ("X", 11), "F12": ("Y", 12), "F13": ("Z", 13)}
_CAL = {"F14": "X", "F15": "Y", "F16": "Z"}


class SimulatedFarmduino:
    def __init__(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.locked = False
        self.config_ok = True
        self.pins: dict[int, float] = {}
        self.params: dict[int, float] = {2: 1.0}
        self.endstops = {"XA": 0, "XB": 0, "YA": 0, "YB": 0, "ZA": 0, "ZB": 0}
        self.version = "sim.0.1"

    def process(self, line: str) -> list[str]:
        raw = line.strip()
        if not raw:
            return []
        reports = [f"R08 {raw}"]
        try:
            cmd = parse_command(raw)
        except ValueError:
            reports.append("R09")
            return reports
        if self.locked and cmd.code != "F09" and cmd.kind != "E":
            reports.append("R87")
            return reports
        if not self.config_ok and cmd.code.startswith("G"):
            reports.append("R88")
            reports.append("R03 V15")
            return reports
        reports.append("R01")
        reports.extend(self._run(cmd))
        if not any(r.startswith("R03") or r.startswith("R09") for r in reports):
            reports.append("R02")
        return reports

    def _run(self, cmd: Command) -> list[str]:
        code = cmd.code
        p = cmd.params
        if cmd.kind == "E":
            self.locked = True
            return ["R87"]
        if code == "@":
            return ["R86"]
        if code == "F09":
            self.locked = False
            return []
        if code == "G00":
            if "X" in p:
                self.x = p["X"]
            if "Y" in p:
                self.y = p["Y"]
            if "Z" in p:
                self.z = p["Z"]
            return [self._r82()]
        if code == "G28":
            self.x = self.y = self.z = 0.0
            return ["R11", "R12", "R13", self._r82()]
        if code in _HOME:
            axis, rn = _HOME[code]
            setattr(self, axis.lower(), 0.0)
            return [f"R{rn}", self._r82()]
        if code in _CAL:
            axis = _CAL[code]
            setattr(self, axis.lower(), 0.0)
            return [f"R06 {axis}0", self._r82()]
        if code == "F20":
            lines = [encode("R21", P=k, V=v) for k, v in sorted(self.params.items())]
            lines.append("R20")
            return lines
        if code == "F21":
            pin = int(p.get("P", 0))
            return [encode("R21", P=pin, V=self.params.get(pin, 0))]
        if code == "F22":
            pin = int(p.get("P", 0))
            self.params[pin] = p.get("V", 0)
            if pin == 2:
                self.config_ok = bool(self.params[pin])
            return [encode("R21", P=pin, V=self.params[pin])]
        if code == "F41":
            pin = int(p.get("P", 0))
            self.pins[pin] = p.get("V", 0)
            return [encode("R41", P=pin, V=self.pins[pin])]
        if code == "F42":
            pin = int(p.get("P", 0))
            return [encode("R41", P=pin, V=self.pins.get(pin, 0))]
        if code == "F43":
            return []
        if code == "F44":
            pin = int(p.get("P", 0))
            self.pins[pin] = p.get("W", 0)
            return [encode("R41", P=pin, V=self.pins[pin])]
        if code == "F61":
            pin = int(p.get("P", 0))
            self.pins[pin] = p.get("V", 0)
            return [encode("R41", P=pin, V=self.pins[pin])]
        if code == "F81":
            return ["R81 " + " ".join(f"{k}{v}" for k, v in self.endstops.items())]
        if code == "F82":
            return [self._r82()]
        if code == "F83":
            return [f"R83 {self.version}"]
        if code == "F84":
            if p.get("X"):
                self.x = 0.0
            if p.get("Y"):
                self.y = 0.0
            if p.get("Z"):
                self.z = 0.0
            return [self._r82()]
        if code in {"F01", "F02", "F51", "F52"}:
            return []
        return ["R09"]

    def _r82(self) -> str:
        return encode("R82", X=self.x, Y=self.y, Z=self.z)
