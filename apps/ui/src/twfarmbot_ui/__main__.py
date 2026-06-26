"""Entry point: ``twfarmbot-ui`` runs the Streamlit dashboard."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    from twfarmbot_core.logging import configure_logging

    configure_logging()

    app_path = Path(__file__).parent / "app.py"
    # Streamlit expects the form: streamlit run path/to/app.py [args]
    sys.argv = ["streamlit", "run", str(app_path), *sys.argv[1:]]
    port = os.getenv("TWFB_UI_PORT", "8501")
    if "--server.port" not in sys.argv:
        sys.argv += ["--server.port", port]
    os.execvp("streamlit", sys.argv)


if __name__ == "__main__":
    main()
