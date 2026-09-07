from __future__ import annotations

import logging
import os
import time

from twfarmbot_core.logging import configure_logging, get_logger

from .inspect_job import inspect_config, run_autonomous_loop, run_once


def _run_worker_cycle() -> None:
    mode = os.getenv("TWFB_WORKER_MODE", "autonomous").lower()
    try:
        if mode == "autonomous":
            run_autonomous_loop()
        else:
            run_once()
    except Exception:  # noqa: BLE001
        logging.getLogger("twfarmbot.worker").exception("scheduled worker cycle failed")


def main() -> None:
    configure_logging()
    log = get_logger("twfarmbot.worker")
    cfg = inspect_config()
    interval = cfg["interval_s"]
    log.info(
        "Starting TWFarmBot worker (inspect_interval_s=%s api=%s)",
        interval,
        cfg["api_url"],
    )
    if interval <= 0:
        log.info("scheduled inspect disabled (interval_s<=0)")
        while True:
            time.sleep(3600)
    if os.getenv("TWFB_INSPECT_ON_START", "0").lower() in {"1", "true", "yes"}:
        _run_worker_cycle()
    while True:
        log.info("next inspect in %ss", interval)
        time.sleep(interval)
        _run_worker_cycle()


if __name__ == "__main__":
    main()
