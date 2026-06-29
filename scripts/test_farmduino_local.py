#!/usr/bin/env python3
"""Local hardware smoke test for the Genesis v1.8 / Farmduino serial stack.

Run with the Farmduino powered and connected via USB (usually /dev/ttyACM0).
By default the script only reads state and does NOT move anything or turn on
peripherals.  Use the flags below only when it is safe to do so.

Examples:
    # Read-only checks (position, endstops, firmware params)
    PYTHONPATH= uv run python scripts/test_farmduino_local.py

    # Home all axes
    PYTHONPATH= uv run python scripts/test_farmduino_local.py --home

    # Move 10 mm up in Z
    PYTHONPATH= uv run python scripts/test_farmduino_local.py --move 0 0 10

    # Toggle the water solenoid for 0.5 s
    PYTHONPATH= uv run python scripts/test_farmduino_local.py --water 0.5
"""

from __future__ import annotations

import argparse
import logging
import sys

from farmbot_serial import FarmduinoSerial
from watering_service import get_backend

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("farmduino_smoke")


def _serial_smoke(args: argparse.Namespace) -> int:
    """Use the low-level FarmduinoSerial driver."""
    client = FarmduinoSerial()
    try:
        client.connect()
    except Exception as err:  # noqa: BLE001
        log.error("could not open Farmduino serial port: %s", err)
        return 1

    try:
        log.info("firmware steps/mm: %s", client.steps_per_mm())
        log.info("position: %s", client.get_position())
        log.info("endstops: %s", client.get_endstops())

        if args.home:
            log.info("homing all axes...")
            client.home("all")
            log.info("homed; position: %s", client.get_position())

        if args.move:
            x, y, z = args.move
            log.info("moving to x=%s y=%s z=%s...", x, y, z)
            client.move(x, y, z)
            log.info("move done; position: %s", client.get_position())

        if args.read_pin is not None:
            value = client.read_pin(args.read_pin, mode=args.pin_mode)
            log.info("pin %s (%s) = %s", args.read_pin, args.pin_mode, value)

        if args.write_pin is not None:
            pin, value = args.write_pin
            log.info("writing pin %s = %s (%s)", pin, value, args.pin_mode)
            client.write_pin(pin, value, mode=args.pin_mode)

        if args.water:
            log.info("pulsing water solenoid for %s s...", args.water)
            client.write_pin_timed(8, 1, seconds=args.water)
    finally:
        client.close()

    return 0


def _backend_smoke(args: argparse.Namespace) -> int:
    """Use the high-level backend that the API server talks to."""
    backend = get_backend()
    try:
        backend.connect()
    except Exception as err:  # noqa: BLE001
        log.error("backend connect failed: %s", err)
        return 1

    log.info("backend status: %s", backend.get_status())
    log.info("cached xyz: %s", backend.get_xyz())

    if args.home:
        log.info("backend: homing...")
        backend.find_home()
        log.info("backend: position after home: %s", backend.get_xyz())

    if args.move:
        x, y, z = args.move
        log.info("backend: moving to x=%s y=%s z=%s...", x, y, z)
        backend.move(x, y, z)
        log.info("backend: position after move: %s", backend.get_xyz())

    if args.water:
        log.info("backend: watering for %s s...", args.water)
        backend.water(args.water)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        action="store_true",
        help="test through the watering_service backend instead of the raw serial driver",
    )
    parser.add_argument(
        "--home",
        action="store_true",
        help="home all axes (only if the bed is clear)",
    )
    parser.add_argument(
        "--move",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="move to absolute coordinates in mm",
    )
    parser.add_argument(
        "--water",
        type=float,
        metavar="SECONDS",
        help="pulse the water solenoid for the given duration",
    )
    parser.add_argument(
        "--read-pin",
        type=int,
        metavar="PIN",
        help="read a pin and print the value",
    )
    parser.add_argument(
        "--write-pin",
        nargs=2,
        type=int,
        metavar=("PIN", "VALUE"),
        help="write a digital value to a pin",
    )
    parser.add_argument(
        "--pin-mode",
        default="digital",
        choices=("digital", "analog"),
        help="pin mode for --read-pin/--write-pin",
    )
    args = parser.parse_args()

    if args.backend:
        return _backend_smoke(args)
    return _serial_smoke(args)


if __name__ == "__main__":
    sys.exit(main())
