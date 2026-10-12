#!/usr/bin/env bash
source "$HOME/ComfyUI/venv/bin/activate"
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_ROCM_ARCH=gfx1030
rm -f dp4a_kernels.hip
exec python test_linear.py "$@"
