from __future__ import annotations

import logging
import os
import time

from twfarmbot_core.logging import configure_logging, get_logger

from .inspect_job import inspect_config, run_once


def _run_inspect() -> None:
    try:
        run_once()
    except Exception:  # noqa: BLE001
        logging.getLogger("twfarmbot.worker").exception("scheduled inspect failed")


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
        _run_inspect()
    while True:
        log.info("next inspect in %ss", interval)
        time.sleep(interval)
        _run_inspect()


if __name__ == "__main__":
    main()
