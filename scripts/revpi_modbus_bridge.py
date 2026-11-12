"""Bridge the PiCtory ModbusTCP Slave to physical DIO on the RevPi.

Architecture (verified on the RevPi 2026-11-10): the slave's Modbus memory
is NOT synced into the piControl process image -- holding registers read and
write fine over TCP but never appear as ``Output_*``/``Input_*`` in the
image, even with all Export flags set and after reboot. So instead of
watching the image, this loop polls the slave over local TCP
(``127.0.0.1:502``, holding register 0, function 3) with a tiny stdlib-only
client and copies bit 0 to the DIO relay (``O_1``).

Our ``tool`` on/off writes land in that same register, so the chain is:
API -> RevPi:502 register 0 -> this loop -> DIO relay. Status is the
commanded word read back by the API (holding register 0): the slave's
input words are read-only from the RevPi side (revpimodio2 sets
``_read_only_io`` for everything but outputs), so with no feedback wire
commanded state is the honest signal.

Run on the RevPi (revpimodio2 ships on the Revolution Pi image, no pip
needed)::

    python3 scripts/revpi_modbus_bridge.py --list
    sudo cp scripts/systemd/revpi-modbus-bridge.service /etc/systemd/system/
    sudo systemctl enable --now revpi-modbus-bridge
"""

from __future__ import annotations

import argparse
import logging
import socket
import struct
import time

# PiCtory value names / Modbus map.
MODBUS_HOST = "127.0.0.1"
MODBUS_PORT = 502
MODBUS_UNIT = 1
MODBUS_REGISTER = 0
DIO_OUT = "O_1"

POLL_S = 0.05

log = logging.getLogger("revpi_modbus_bridge")


class ModbusError(Exception):
    """The slave answered with an exception, or the frame was bad."""


def _recvn(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ModbusError("connection closed mid-frame")
        buf += chunk
    return buf


def read_holding_register(host: str, port: int, address: int, unit: int, timeout: float) -> int:
    """Read one holding register (Modbus TCP, function 3). Stdlib only."""
    req = struct.pack(">HHHBBHH", 0x0001, 0x0000, 6, unit, 3, address, 1)
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall(req)
        _tid, _pid, length, _runit, func = struct.unpack(">HHHBB", _recvn(sock, 8))
        body = _recvn(sock, length - 2)  # length counts unit + function bytes
        if func & 0x80:
            raise ModbusError(f"slave exception {body[0]}")
        if func != 3 or len(body) < 3 or body[0] != 2:
            raise ModbusError(f"bad response function={func}")
        return struct.unpack(">H", body[1:3])[0]


def _names(rpi) -> list[str]:  # noqa: ANN001
    return sorted(io.name for io in rpi.io)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="print IO names and exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import revpimodio2

    rpi = revpimodio2.RevPiModIO(autorefresh=True)
    if args.list:
        print("\n".join(_names(rpi)))
        return
    try:
        dio_out = rpi.io[DIO_OUT]
    except KeyError:
        raise SystemExit(f"IO {DIO_OUT!r} not found. Available: {', '.join(_names(rpi))}") from None

    log.info("bridging %s:%d register %d bit0 -> %s", MODBUS_HOST, MODBUS_PORT, MODBUS_REGISTER, DIO_OUT)
    warned = False
    while True:
        try:
            word = read_holding_register(MODBUS_HOST, MODBUS_PORT, MODBUS_REGISTER, MODBUS_UNIT, 2.0)
            dio_out.value = bool(word & 1)
            warned = False
        except (OSError, ModbusError) as err:
            if not warned:
                log.warning("slave/reg unreachable (%s); retrying", err)
                warned = True
        except RuntimeError as err:
            # e.g. DIO module physically absent: stay alive, keep retrying.
            if not warned:
                log.warning("cannot drive %s (%s); retrying", DIO_OUT, err)
                warned = True
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
