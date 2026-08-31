#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_SYSTEMD_DIR="${HOME}/.config/systemd/user"

mkdir -p "${USER_SYSTEMD_DIR}"
cp "${SCRIPT_DIR}/systemd/"*.service "${USER_SYSTEMD_DIR}/"

systemctl --user daemon-reload
systemctl --user enable twfarmbot-resireg twfarmbot-api twfarmbot-ui twfarmbot-worker

echo "TWFarmBot user services installed and enabled."
echo ""
echo "To start on boot before login, run once as root:"
echo "  sudo loginctl enable-linger ${USER}"
echo ""
echo "To start now (systemd):"
echo "  systemctl --user start twfarmbot-resireg twfarmbot-api twfarmbot-ui twfarmbot-worker"
echo ""
echo "For a local uv process set instead:"
echo "  ./scripts/start_all.sh"
