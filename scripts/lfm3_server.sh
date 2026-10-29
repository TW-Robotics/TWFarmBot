#!/usr/bin/env bash
# Local LFM3 model server (llama.cpp, OpenAI-compatible) for the Assistant's "Local" provider.
#   ./scripts/lfm3_server.sh                      # LFM3-Omni-2B: text, tools, images, audio
#   LFM3_MODEL=LFM3-1.7B ./scripts/lfm3_server.sh # text only
#   LFM3_BACKEND=cpu|vulkan|cuda                  # default: vulkan when an NVIDIA GPU works
# Expects llama.cpp release builds (b11940 or newer) and GGUF files under $LFM3_HOME
# (default ~/lfm3): bin/ (CPU), bin-vulkan/, bin-cuda/. The CUDA 12.8 build needs a
# driver newer than 535, which is why Vulkan is the default here.
set -euo pipefail

LFM3_HOME="${LFM3_HOME:-$HOME/lfm3}"
MODEL="${LFM3_MODEL:-LFM3-Omni-2B}"
BACKEND="${LFM3_BACKEND:-auto}"
if [[ "$BACKEND" == auto ]]; then
  if nvidia-smi -L >/dev/null 2>&1 && [[ -x "$LFM3_HOME/bin-vulkan/llama-server" ]]; then
    BACKEND=vulkan
  else
    BACKEND=cpu
  fi
fi
BIN="$LFM3_HOME/bin"
[[ "$BACKEND" != cpu ]] && BIN="$LFM3_HOME/bin-$BACKEND"

ARGS=(-m "$LFM3_HOME/models/$MODEL-Q4_K_M.gguf" --alias lfm3 --host 127.0.0.1
  --port "${LFM3_PORT:-8090}" -c "${LFM3_CTX:-32768}" --jinja -t "${LFM3_THREADS:-8}"
  # Liquid's recommended sampling for chat/tools, and a hard cap so a stuck model stops.
  --top-k 50 --repeat-penalty 1.1 -n "${LFM3_MAX_TOKENS:-1536}")
MMPROJ="$LFM3_HOME/models/mmproj-$MODEL-Q8_0.gguf"
[[ -f "$MMPROJ" ]] && ARGS+=(--mmproj "$MMPROJ")
if [[ "$BACKEND" != cpu ]]; then
  ARGS+=(-ngl 99)
  # Machines with an integrated GPU list it first; pin the NVIDIA one.
  DEVICE=$("$BIN/llama-server" --list-devices 2>/dev/null | awk '/NVIDIA/ {sub(":", "", $1); print $1; exit}')
  [[ -n "$DEVICE" ]] && ARGS+=(--device "$DEVICE")
fi
exec "$BIN/llama-server" "${ARGS[@]}"
