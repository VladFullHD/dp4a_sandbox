import os
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx1030")

from benchmark import load_ext
from dp4a_linear import DP4ALinear

ext = load_ext()

def print_metrics(name, out_test, out_ref):
    l2_err = (torch.linalg.vector_norm(out_test - out_ref) / torch.linalg.vector_norm(out_ref)).item()
    cos_sim = F.cosine_similarity(out_test.flatten(), out_ref.flatten(), dim=0).item()
    max_err = (out_test - out_ref).abs().max().item()
    print(f"[{name}]")
    print(f"  • Относительная L2-ошибка:      {l2_err:.6f} ({l2_err*100:.3f}%)")
    print(f"  • Cosine Similarity (сходство): {cos_sim:.6f} (идеал = 1.0)")
    print(f"  • Макс. абсолютная ошибка:        {max_err:.6f}")

def bench_median(fn, iters=20):
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2], times[0], times[-1]

# -----------------------------------------------------------------
# 1. ТЕСТ 4096x4096 (Честное сравнение с идентичными весами)
# -----------------------------------------------------------------
size = 4096
linear_fp32 = nn.Linear(size, size, bias=True).cuda()

# Синхронизируем веса FP16 строго с FP32!
linear_fp16 = nn.Linear(size, size, bias=True).half().cuda()
linear_fp16.weight.data.copy_(linear_fp32.weight.data.half())
linear_fp16.bias.data.copy_(linear_fp32.bias.data.half())

linear_dp4a = DP4ALinear(linear_fp32, ext)

x_fp32 = torch.randn(size, size, device="cuda", dtype=torch.float32)
# ВЫНОСИМ HALF ДО ТАЙМЕРА:
x_fp16 = x_fp32.half()

with torch.inference_mode():
    for _ in range(10):
        _ = linear_fp32(x_fp32)
        _ = linear_fp16(x_fp16)
        _ = linear_dp4a(x_fp32)
    torch.cuda.synchronize()

    fp32_med, fp32_min, fp32_max = bench_median(lambda: linear_fp32(x_fp32))
    fp16_med, fp16_min, fp16_max = bench_median(lambda: linear_fp16(x_fp16))
    dp4a_med, dp4a_min, dp4a_max = bench_median(lambda: linear_dp4a(x_fp32))

print("\n" + "="*70)
print(f" ТЕСТ 1: КВАДРАТНЫЙ СЛОЙ ({size}x{size}) — ЧЕСТНАЯ МЕДИАНА 20 ЗАПУСКОВ")
print("="*70)
print(f"FP32 rocBLAS:  {fp32_med:.3f} ms (min: {fp32_min:.3f}, max: {fp32_max:.3f})")
print(f"FP16 rocBLAS:  {fp16_med:.3f} ms (min: {fp16_min:.3f}, max: {fp16_max:.3f})")
print(f"DP4ALinear:    {dp4a_med:.3f} ms (min: {dp4a_min:.3f}, max: {dp4a_max:.3f})")
print("-" * 70)
print(f"Ускорение относительно FP32: {fp32_med / dp4a_med:.3f}x")
print(f"Преимущество над FP16:       {fp16_med / dp4a_med:.3f}x")
print_metrics("Точность 4096x4096", linear_dp4a(x_fp32), linear_fp32(x_fp32))

# -----------------------------------------------------------------
# 2. ТЕСТ SDXL [2, 77, 2048] -> [2, 77, 1280]
# -----------------------------------------------------------------
print("\n" + "="*70)
print(" ТЕСТ 2: SDXL [2, 77, 2048] -> [2, 77, 1280]")
print("="*70)
sdxl_l_fp32 = nn.Linear(2048, 1280, bias=True).cuda()
sdxl_l_dp4a = DP4ALinear(sdxl_l_fp32, ext)
x_sdxl = torch.randn(2, 77, 2048, device="cuda", dtype=torch.float32)

with torch.inference_mode():
    for _ in range(5):
        _ = sdxl_l_fp32(x_sdxl)
        _ = sdxl_l_dp4a(x_sdxl)
    torch.cuda.synchronize()

    sdxl_fp32_med, _, _ = bench_median(lambda: sdxl_l_fp32(x_sdxl))
    sdxl_dp4a_med, _, _ = bench_median(lambda: sdxl_l_dp4a(x_sdxl))

    out_ref = sdxl_l_fp32(x_sdxl)
    out_test = sdxl_l_dp4a(x_sdxl)

print(f"Медиана FP32: {sdxl_fp32_med:.3f} ms | Медиана DP4A: {sdxl_dp4a_med:.3f} ms (Ускорение: {sdxl_fp32_med/sdxl_dp4a_med:.3f}x)")
print_metrics("Точность SDXL 3D", out_test, out_ref)

# -----------------------------------------------------------------
# 3. ГРАНИЧНЫЙ ТЕСТ M=1 (СПЕЦИАЛИЗИРОВАННОЕ GEMV ЯДРО)
# -----------------------------------------------------------------
print("\n" + "="*70)
print(" ТЕСТ 3: ОДИНОЧНЫЙ ТОКЕН M = 1 [1, 2048] -> [1, 1280] (GEMV РЕДУКЦИЯ)")
print("="*70)
x_single = torch.randn(1, 2048, device="cuda", dtype=torch.float32)
with torch.inference_mode():
    single_fp32_med, _, _ = bench_median(lambda: sdxl_l_fp32(x_single))
    single_dp4a_med, _, _ = bench_median(lambda: sdxl_l_dp4a(x_single))
    out_single_ref = sdxl_l_fp32(x_single)
    out_single_test = sdxl_l_dp4a(x_single)

print(f"Медиана FP32: {single_fp32_med:.3f} ms | Медиана DP4A GEMV: {single_dp4a_med:.3f} ms")
print_metrics("Точность M = 1", out_single_test, out_single_ref)

# -----------------------------------------------------------------
# 4. ТЕСТ ПУСТОГО ТЕНЗОРА (ПРОВЕРКА КОНТРАКТА PYTORCH)
# -----------------------------------------------------------------
print("\n" + "="*70)
print(" ТЕСТ 4: ПУСТОЙ ТЕНЗОР [0, 2048] -> [0, 1280]")
print("="*70)
x_empty = torch.randn(0, 2048, device="cuda", dtype=torch.float32)
with torch.inference_mode():
    out_empty = sdxl_l_dp4a(x_empty)
assert out_empty.shape == (0, 1280), f"Ошибка контракта пустого тензора: {out_empty.shape}"
print("УСПЕХ: Пустой вход корректно вернул форму [0, 1280]!")
print("="*70 + "\n")