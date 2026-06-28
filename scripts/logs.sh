#!/usr/bin/env bash
set -euo pipefail

journalctl --user -u twfarmbot-resireg -u twfarmbot-api -u twfarmbot-ui -f
