#!/usr/bin/env bash
set -euo pipefail

systemctl --user stop twfarmbot-ui twfarmbot-api twfarmbot-resireg
echo "TWFarmBot services stopped."
