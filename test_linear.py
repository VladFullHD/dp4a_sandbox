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

# Создаем слои
linear_fp32 = nn.Linear(size, size, bias=True).cuda()
linear_fp16 = nn.Linear(size, size, bias=True).half().cuda()
linear_dp4a = DP4ALinear(linear_fp32, ext).cuda()

x_fp32 = torch.randn(size, size, device="cuda", dtype=torch.float32)
x_fp16 = x_fp32.half()

# Включаем строгий режим инференса (отключает трекинг autograd у всех слоев!)
with torch.inference_mode():
    # Прогрев GPU
    for _ in range(10):
        _ = linear_fp32(x_fp32)
        _ = linear_fp16(x_fp16)
        _ = linear_dp4a(x_fp32)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # 1. Замер FP32 rocBLAS
    start_event.record()
    for _ in range(iters):
        _ = linear_fp32(x_fp32)
    end_event.record()
    torch.cuda.synchronize()
    fp32_ms = start_event.elapsed_time(end_event) / iters

    # 2. Замер FP16 rocBLAS
    start_event.record()
    for _ in range(iters):
        _ = linear_fp16(x_fp16)
    end_event.record()
    torch.cuda.synchronize()
    fp16_ms = start_event.elapsed_time(end_event) / iters

    # 3. Замер DP4ALinear с FUSED EPILOGUE
    start_event.record()
    for _ in range(iters):
        _ = linear_dp4a(x_fp32)
    end_event.record()
    torch.cuda.synchronize()
    dp4a_ms = start_event.elapsed_time(end_event) / iters

    out_fp32 = linear_fp32(x_fp32)
    out_dp4a = linear_dp4a(x_fp32)
    error = (torch.linalg.vector_norm(out_dp4a - out_fp32) / torch.linalg.vector_norm(out_fp32)).item()

print("\n" + "="*65)
print(f" ЧЕСТНЫЙ ТРЕХСТОРОННИЙ БЕНЧМАРК В INFERENCE_MODE ({size}x{size})")
print("="*65)
print(f"1. torch.nn.Linear (FP32 rocBLAS): {fp32_ms:.3f} ms")
print(f"2. torch.nn.Linear (FP16 rocBLAS): {fp16_ms:.3f} ms")
print(f"3. DP4ALinear (FUSED EPILOGUE):    {dp4a_ms:.3f} ms")
print("-" * 65)
print(f"Ускорение DP4A относительно FP32:   {fp32_ms / dp4a_ms:.3f}x")
print(f"Соотношение с нативным FP16:       {fp16_ms / dp4a_ms:.3f}x")
print(f"Относительная L2 погрешность:      {error:.6f}")
print("="*65 + "\n")