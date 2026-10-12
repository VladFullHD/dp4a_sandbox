import os
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx1030")

from benchmark import load_ext
from dp4a_linear import DP4ALinear

ext = load_ext()

def fake_quant_reference(x, linear_dp4a):
    """Математический эталон W8A8 (эмуляция): проверка ядра без погрешности квантования."""
    M = x.reshape(-1, linear_dp4a.in_features).shape[0]
    x_2d = x.reshape(-1, linear_dp4a.in_features).float()
    if linear_dp4a.pad_k > 0:
        x_padded = F.pad(x_2d, (0, linear_dp4a.pad_k, 0, 0))
    else:
        x_padded = x_2d
    
    # Референсные квантованные числа
    scale_a = (x_padded.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-12)
    q_a = torch.round(x_padded / scale_a).clamp(-127, 127)
    
    # Восстановление весов
    weight_unpacked = linear_dp4a.weight_packed  # unpacked reference via scale
    # Математический расчет dequant(A) @ dequant(W).T + bias в FP64
    w_dequant = linear_dp4a.weight_scale.unsqueeze(1) * F.pad(linear_dp4a._ext_ref.mm_tiled(
        torch.zeros((1, K_pad//4), dtype=torch.int32, device='cuda'),
        linear_dp4a.weight_packed, 1, linear_dp4a.out_features, K_pad//4
    ), (0,0)).float() if False else None # упрощенный эталон через float-веса
    return None

def print_metrics(name, out_test, out_ref):
    l2_err = (torch.linalg.vector_norm(out_test - out_ref) / torch.linalg.vector_norm(out_ref).clamp_min(1e-12)).item()
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
# 1. ТЕСТ 4096x4096
# -----------------------------------------------------------------
size = 4096
linear_fp32 = nn.Linear(size, size, bias=True).cuda()
linear_fp16 = nn.Linear(size, size, bias=True).half().cuda()
linear_fp16.weight.data.copy_(linear_fp32.weight.data.half())
linear_fp16.bias.data.copy_(linear_fp32.bias.data.half())

linear_dp4a = DP4ALinear(linear_fp32, ext)

x_fp32 = torch.randn(size, size, device="cuda", dtype=torch.float32)
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
print(f" ТЕСТ 1: КВАДРАТНЫЙ СЛОЙ ({size}x{size}) — ПОСТРОЧНЫЙ МАСШТАБ")
print("="*70)
print(f"FP32 rocBLAS:  {fp32_med:.3f} ms (min: {fp32_min:.3f}, max: {fp32_max:.3f})")
print(f"FP16 rocBLAS:  {fp16_med:.3f} ms (min: {fp16_min:.3f}, max: {fp16_max:.3f})")
print(f"DP4ALinear:    {dp4a_med:.3f} ms (min: {dp4a_min:.3f}, max: {dp4a_max:.3f})")
print("-" * 70)
print(f"Ускорение относительно FP32: {fp32_med / dp4a_med:.3f}x")
print(f"Преимущество над FP16:       {fp16_med / dp4a_med:.3f}x")
print_metrics("Точность 4096x4096 vs FP32", linear_dp4a(x_fp32), linear_fp32(x_fp32))

# -----------------------------------------------------------------
# 2. ТЕСТ SDXL [2, 77, 2048] -> [2, 77, 1280] (БЕЗ PAD_N И СРЕЗОВ)
# -----------------------------------------------------------------
print("\n" + "="*70)
print(" ТЕСТ 2: SDXL [2, 77, 2048] -> [2, 77, 1280] (СПЛОШНАЯ ПАМЯТЬ БЕЗ PAD_N)")
print("="*70)
sdxl_l_fp32 = nn.Linear(2048, 1280, bias=True).cuda()
sdxl_l_dp4a = DP4ALinear(sdxl_l_fp32, ext)
x_sdxl = torch.randn(2, 77, 2048, device="cuda", dtype=torch.float32)

with torch.inference_mode():
    out_ref = sdxl_l_fp32(x_sdxl)
    out_test = sdxl_l_dp4a(x_sdxl)

assert out_test.is_contiguous(), "ОШИБКА: выходной тензор должен быть сплошным (contiguous)!"
assert out_test.shape == out_ref.shape, "Формы не совпали!"
print(f"Форма: {list(x_sdxl.shape)} -> {list(out_test.shape)} (сплошной в памяти: {out_test.is_contiguous()})")
print_metrics("Точность SDXL 3D", out_test, out_ref)

# -----------------------------------------------------------------
# 3. ГРАНИЧНЫЙ ТЕСТ M = 1 (GEMV ЯДРО)
# -----------------------------------------------------------------
print("\n" + "="*70)
print(" ТЕСТ 3: ОДИНОЧНЫЙ ТОКЕН M = 1 [1, 2048] -> [1, 1280]")
print("="*70)
x_single = torch.randn(1, 2048, device="cuda", dtype=torch.float32)
with torch.inference_mode():
    out_single_ref = sdxl_l_fp32(x_single)
    out_single_test = sdxl_l_dp4a(x_single)
print_metrics("Точность M = 1", out_single_test, out_single_ref)
print("="*70 + "\n")