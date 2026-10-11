# DP4A sandbox for Radeon RX 6750 XT / ROCm

This is an isolated research prototype. It does not import, edit, or patch ComfyUI.

## What is included

- `dp4a_ext.cpp` / `dp4a_kernels.cu`: PyTorch JIT extension using AMD's signed 4xINT8 dot-product builtin (`__builtin_amdgcn_sdot4`), a naive output-per-thread kernel, and a 32x32 tiled kernel using HIP shared/LDS memory.
- `benchmark.py`: correctness test against exact int32 arithmetic on small matrices, FP32/FP16 `torch.matmul` timings, tiled 4096x4096 DP4A timing, approximation error after dequantization, and a naive-kernel timing.
- `quantization.py`: affine min/max quantization and unpack helper. Min/max reduction uses PyTorch's GPU reductions; the custom kernel does quantize+pack four signed bytes into each int32.
- `dp4a_linear.py`: isolated `DP4ALinear` prototype. It is **not integrated into ComfyUI** and should not be used on a production workflow before accuracy and performance validation.

## Install/run on the user's Fedora ROCm environment

Use the same Python environment used by ComfyUI, with ComfyUI stopped (to reduce concurrent GPU memory use):

```bash
cd ~/dp4a_sandbox
source ~/ComfyUI/venv/bin/activate
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_ROCM_ARCH=gfx1030
export MAX_JOBS=2
python benchmark.py
```

First run JIT-compiles the extension, which can take several minutes. HIP compilation through PyTorch's `cpp_extension.load` can vary between ROCm/PyTorch releases; if compilation fails, preserve the full compiler output. No claim is made that this project has been compiled or run on the user's machine.

The first benchmark defaults to comparing FP32/FP16 matmul and the tiled DP4A kernel at 4096x4096. To protect a desktop using the same GPU, the highly serial naive kernel defaults to 512x512. To measure naive DP4A at 4096, explicitly run:

```bash
python benchmark.py --naive-size 4096
```

That can occupy the display GPU for a long time. For a quicker validation run:

```bash
python benchmark.py --size 512 --naive-size 128 --iters 3
```

## Important technical constraints

- `gfx1030` is the architecture override used by the user's ROCm environment. The GPU is a Radeon RX 6750 XT (RDNA 2); the `v_dot4_i32_i8` instruction is listed for the GFX1030 ISA. Confirm generated code and correctness on the actual machine.
- DP4A is integer arithmetic. Inputs are quantized to signed INT8; dequantized outputs are approximate, not bit-identical to FP32 matmul. The exact-int32 correctness check proves the dot-product instruction and packing layout, not the floating-point approximation quality.
- The 32x32 tile is an experimental baseline, not guaranteed to beat rocBLAS/PyTorch. The FP16/FP32 matmul paths may be highly optimized, so compare measured timings.
- The naive 4096 kernel is intentionally opt-in because the RX 6750 XT may also drive the desktop.
- The affine quantizer currently uses `torch.amin`/`torch.amax` for GPU reductions, followed by the custom packing kernel. It is not yet fused into GEMM.
- `DP4ALinear` uses one symmetric scale for each weight matrix and one scale per input tensor. This is a simple accuracy baseline, not production quantization. SDXL/ComfyUI has many activation distributions, attention shapes, LoRA patches, and mixed-dtype assumptions; global module replacement is deliberately omitted until the microbenchmark is correct and there is evidence of a speedup.

## Suggested staged progression

1. Run correctness checks and save FP32/FP16/DP4A timings.
2. Tune tile dimensions and compiler output after verifying exact int32 agreement.
3. Fuse quantization/packing with activation preprocessing and handle asymmetric zero-point correction if using affine quantization for GEMM.
4. Only then test `DP4ALinear` in a standalone model layer. Integrating into ComfyUI's ModelPatcher is a separate compatibility task and should not be attempted until per-layer error and end-to-end image quality are measured.
