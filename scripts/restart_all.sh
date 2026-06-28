#!/usr/bin/env bash
set -euo pipefail

systemctl --user restart twfarmbot-resireg twfarmbot-api twfarmbot-ui
echo "TWFarmBot services restarted."
systemctl --user status twfarmbot-resireg twfarmbot-api twfarmbot-ui --no-pager
