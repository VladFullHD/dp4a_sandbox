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
    print(f"  • Относительная L2-ошибка:   {l2_err:.6f} ({l2_err*100:.3f}%)")
    print(f"  • Cosine Similarity (сходство): {cos_sim:.6f} (идеал = 1.0)")
    print(f"  • Макс. абсолютная ошибка:     {max_err:.6f}")

# -----------------------------------------------------------------
# 1. ТЕСТ 4096x4096
# -----------------------------------------------------------------
size = 4096
iters = 20
linear_fp32 = nn.Linear(size, size, bias=True).cuda()
linear_fp16 = nn.Linear(size, size, bias=True).half().cuda()
linear_dp4a = DP4ALinear(linear_fp32, ext)

x = torch.randn(size, size, device="cuda", dtype=torch.float32)

with torch.inference_mode():
    for _ in range(10):
        _ = linear_fp32(x)
        _ = linear_fp16(x.half())
        _ = linear_dp4a(x)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters): _ = linear_fp32(x)
    end.record(); torch.cuda.synchronize()
    fp32_ms = start.elapsed_time(end) / iters

    start.record()
    for _ in range(iters): _ = linear_fp16(x.half())
    end.record(); torch.cuda.synchronize()
    fp16_ms = start.elapsed_time(end) / iters

    start.record()
    for _ in range(iters): _ = linear_dp4a(x)
    end.record(); torch.cuda.synchronize()
    dp4a_ms = start.elapsed_time(end) / iters

print("\n" + "="*65)
print(f" РЕЗУЛЬТАТЫ СЛОЯ 4096x4096 (Медиана 20 итераций)")
print("="*65)
print(f"FP32 rocBLAS:  {fp32_ms:.3f} ms")
print(f"FP16 rocBLAS:  {fp16_ms:.3f} ms")
print(f"DP4ALinear:    {dp4a_ms:.3f} ms (Ускорение: {fp32_ms/dp4a_ms:.3f}x vs FP32, {fp16_ms/dp4a_ms:.3f}x vs FP16)")
print_metrics("Точность 4096x4096", linear_dp4a(x), linear_fp32(x))

# -----------------------------------------------------------------
# 2. ТЕСТ SDXL [2, 77, 2048] БЕЗ ХОЛОСТОГО ПАДДИНГА СТРОК
# -----------------------------------------------------------------
print("\n" + "="*65)
print(" ТЕСТ SDXL [2, 77, 2048] -> [2, 77, 1280] (Холостой паддинг M отключен!)")
print("="*65)
sdxl_l_fp32 = nn.Linear(2048, 1280, bias=True).cuda()
sdxl_l_dp4a = DP4ALinear(sdxl_l_fp32, ext)
x_sdxl = torch.randn(2, 77, 2048, device="cuda", dtype=torch.float32)

with torch.inference_mode():
    out_ref = sdxl_l_fp32(x_sdxl)
    out_test = sdxl_l_dp4a(x_sdxl)

assert out_test.shape == out_ref.shape, "Формы не совпали!"
print(f"Форма: {list(x_sdxl.shape)} -> {list(out_test.shape)}")
print_metrics("Точность SDXL 3D", out_test, out_ref)

# -----------------------------------------------------------------
# 3. ГРАНИЧНЫЙ ТЕСТ: M = 1 (Одиночный токен)
# -----------------------------------------------------------------
print("\n" + "="*65)
print(" ГРАНИЧНЫЙ ТЕСТ: M = 1 [1, 2048] -> [1, 1280]")
print("="*65)
x_single = torch.randn(1, 2048, device="cuda", dtype=torch.float32)
with torch.inference_mode():
    out_single_ref = sdxl_l_fp32(x_single)
    out_single_test = sdxl_l_dp4a(x_single)
print_metrics("Точность M = 1", out_single_test, out_single_ref)
print("="*65 + "\n")