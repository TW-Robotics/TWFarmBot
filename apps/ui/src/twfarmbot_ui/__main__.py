"""Entry point: ``twfarmbot-ui`` serves the native Astryx dashboard."""

from __future__ import annotations


def main() -> None:
    from twfarmbot_ui.server import run

    run()


if __name__ == "__main__":
    main()
