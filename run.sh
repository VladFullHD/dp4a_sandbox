#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HOME/ComfyUI/venv/bin/activate"
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_ROCM_ARCH=gfx1030
export MAX_JOBS=2
cd "$ROOT"
exec python benchmark.py "$@"
