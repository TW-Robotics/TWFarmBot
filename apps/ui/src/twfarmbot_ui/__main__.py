from twfarmbot_core.logging import configure_logging, get_logger


def main() -> None:
    configure_logging()
    log = get_logger("twfarmbot.ui")
    log.info("Starting TWFarmBot UI")
    log.info("Dashboard, sensor display and manual triggers live here.")


if __name__ == "__main__":
    main()
