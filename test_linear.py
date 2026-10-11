import os
import time
import torch
import torch.nn as nn

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx1030")

from benchmark import load_ext
from dp4a_linear import DP4ALinear

ext = load_ext()

size = 4096
iters = 20

linear_fp32 = nn.Linear(size, size, bias=True).cuda()
linear_dp4a = DP4ALinear(linear_fp32, ext, kernel="tiled").cuda()

x = torch.randn(size, size, device="cuda", dtype=torch.float32)

# Прогрев GPU (10 итераций)
for _ in range(10):
    _ = linear_fp32(x)
    _ = linear_dp4a(x)
torch.cuda.synchronize()

# -------------------------------------------------------------
# АППАРАТНЫЙ ЗАМЕР GPU (torch.cuda.Event) — Метод, запрошенный коллегой
# -------------------------------------------------------------
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# 1. Замер FP32 через аппаратный таймер чипа
start_event.record()
for _ in range(iters):
    _ = linear_fp32(x)
end_event.record()
torch.cuda.synchronize()
fp32_gpu_ms = start_event.elapsed_time(end_event) / iters

# 2. Замер DP4A End-to-End через аппаратный таймер чипа
start_event.record()
for _ in range(iters):
    _ = linear_dp4a(x)
end_event.record()
torch.cuda.synchronize()
dp4a_gpu_ms = start_event.elapsed_time(end_event) / iters

# Точность
out_fp32 = linear_fp32(x)
out_dp4a = linear_dp4a(x)
error = (torch.linalg.vector_norm(out_dp4a - out_fp32) / torch.linalg.vector_norm(out_fp32)).item()

print("\n" + "="*60)
print(f" АППАРАТНЫЙ ЗАМЕР ЧИПА (torch.cuda.Event) НА {size}x{size}")
print("="*60)
print(f"torch.nn.Linear (FP32 rocBLAS): {fp32_gpu_ms:.3f} ms")
print(f"DP4ALinear (End-to-End честный): {dp4a_gpu_ms:.3f} ms")
print(f"Аппаратное ускорение ядра+слоя:  {fp32_gpu_ms / dp4a_gpu_ms:.3f}x")
print(f"Снижение задержки (Latency):     {(1 - dp4a_gpu_ms / fp32_gpu_ms)*100:.1f}%")
print(f"Относительная L2 погрешность:    {error:.6f}")
print("="*60 + "\n")