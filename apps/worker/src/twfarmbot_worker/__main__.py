from twfarmbot_core.logging import configure_logging, get_logger


def main() -> None:
    configure_logging()
    log = get_logger("twfarmbot.worker")
    log.info("Starting TWFarmBot worker")
    log.info("Background jobs and experiment execution live here.")


if __name__ == "__main__":
    main()
