#!/usr/bin/env bash
set -euo pipefail

systemctl --user start twfarmbot-resireg twfarmbot-api twfarmbot-ui
echo "TWFarmBot services started."
systemctl --user status twfarmbot-resireg twfarmbot-api twfarmbot-ui --no-pager
