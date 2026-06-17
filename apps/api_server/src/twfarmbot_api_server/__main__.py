from twfarmbot_core.config import load_settings
from twfarmbot_core.logging import configure_logging, get_logger


def main() -> None:
    settings = load_settings()
    configure_logging(settings.log_level)
    log = get_logger("twfarmbot.api_server")
    log.info("Starting TWFarmBot API server in env=%s", settings.env)
    log.info("Routes for FarmBot, sensors and experiments live here.")


if __name__ == "__main__":
    main()
