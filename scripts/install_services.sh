#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_SYSTEMD_DIR="${HOME}/.config/systemd/user"

mkdir -p "${USER_SYSTEMD_DIR}"
cp "${SCRIPT_DIR}/systemd/"*.service "${USER_SYSTEMD_DIR}/"

# Pre-accept Streamlit's email prompt so the UI service starts non-interactively.
mkdir -p "${HOME}/.streamlit"
if [[ ! -f "${HOME}/.streamlit/credentials.toml" ]]; then
    cat > "${HOME}/.streamlit/credentials.toml" <<'EOF'
[general]
email = ""
EOF
fi

systemctl --user daemon-reload
systemctl --user enable twfarmbot-resireg twfarmbot-api twfarmbot-ui

echo "TWFarmBot user services installed and enabled."
echo ""
echo "To start on boot before login, run once as root:"
echo "  sudo loginctl enable-linger ${USER}"
echo ""
echo "To start now:"
echo "  ./scripts/start_all.sh"
echo ""
echo "To view logs:"
echo "  ./scripts/logs.sh"
