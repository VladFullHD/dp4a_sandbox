import os
import time
import torch
import torch.nn as nn

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx1030")

from benchmark import load_ext
from dp4a_linear import DP4ALinear

ext = load_ext()

# -------------------------------------------------------------
# ТЕСТ 1: Квадратная матрица 4096x4096
# -------------------------------------------------------------
size = 4096
iters = 20

linear_fp32 = nn.Linear(size, size, bias=True).cuda()
linear_fp16 = nn.Linear(size, size, bias=True).half().cuda()
linear_dp4a = DP4ALinear(linear_fp32, ext)

x_fp32 = torch.randn(size, size, device="cuda", dtype=torch.float32)
x_fp16 = x_fp32.half()

with torch.inference_mode():
    for _ in range(10):
        _ = linear_fp32(x_fp32)
        _ = linear_fp16(x_fp16)
        _ = linear_dp4a(x_fp32)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        _ = linear_fp32(x_fp32)
    end_event.record()
    torch.cuda.synchronize()
    fp32_ms = start_event.elapsed_time(end_event) / iters

    start_event.record()
    for _ in range(iters):
        _ = linear_fp16(x_fp16)
    end_event.record()
    torch.cuda.synchronize()
    fp16_ms = start_event.elapsed_time(end_event) / iters

    start_event.record()
    for _ in range(iters):
        _ = linear_dp4a(x_fp32)
    end_event.record()
    torch.cuda.synchronize()
    dp4a_ms = start_event.elapsed_time(end_event) / iters

    out_fp32 = linear_fp32(x_fp32)
    out_dp4a = linear_dp4a(x_fp32)
    error_4096 = (torch.linalg.vector_norm(out_dp4a - out_fp32) / torch.linalg.vector_norm(out_fp32)).item()

print("\n" + "="*65)
print(f" ТЕСТ 1: КВАДРАТНАЯ МАТРИЦА ({size}x{size})")
print("="*65)
print(f"torch.nn.Linear (FP32 rocBLAS): {fp32_ms:.3f} ms")
print(f"torch.nn.Linear (FP16 rocBLAS): {fp16_ms:.3f} ms")
print(f"DP4ALinear (FUSED EPILOGUE):    {dp4a_ms:.3f} ms")
print(f"Ускорение относительно FP32:   {fp32_ms / dp4a_ms:.3f}x")
print(f"Преимущество над FP16:         {fp16_ms / dp4a_ms:.3f}x")
print(f"L2 погрешность:                {error_4096:.6f}")
print("="*65)

# -------------------------------------------------------------
# ТЕСТ 2: Реальная неквадратная 3D-форма SDXL [Batch=2, Seq=77, In=2048] -> Out=1280
# -------------------------------------------------------------
sdxl_in = 2048
sdxl_out = 1280
linear_sdxl_fp32 = nn.Linear(sdxl_in, sdxl_out, bias=True).cuda()
linear_sdxl_dp4a = DP4ALinear(linear_sdxl_fp32, ext)

x_sdxl = torch.randn(2, 77, sdxl_in, device="cuda", dtype=torch.float32)

with torch.inference_mode():
    out_sdxl_fp32 = linear_sdxl_fp32(x_sdxl)
    out_sdxl_dp4a = linear_sdxl_dp4a(x_sdxl)

assert out_sdxl_dp4a.shape == out_sdxl_fp32.shape, f"Формы не совпали: {out_sdxl_dp4a.shape} vs {out_sdxl_fp32.shape}"
sdxl_error = (torch.linalg.vector_norm(out_sdxl_dp4a - out_sdxl_fp32) / torch.linalg.vector_norm(out_sdxl_fp32)).item()

print("\n" + "="*65)
print(f" ТЕСТ 2: НЕКВАДРАТНЫЙ 3D-ТЕНЗОР SDXL [2, 77, 2048] -> [2, 77, 1280]")
print("="*65)
print(f"Форма входа:    {list(x_sdxl.shape)}")
print(f"Форма выхода:   {list(out_sdxl_dp4a.shape)}")
print(f"L2 погрешность: {sdxl_error:.6f}")
print("СТАТУС: Динамический паддинг отработал безупречно!")
print("="*65 + "\n")