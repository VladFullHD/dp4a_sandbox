import os
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx1030")

from benchmark import load_ext
from dp4a_linear import DP4ALinear

ext = load_ext()

def compute_fake_quant_ref(linear_fp32, linear_dp4a, x):
    """
    Математический эталон W8A8 (эмуляция):
    Считает то же самое квантование, но стандартным PyTorch matmul.
    Если ядро C++ написано без багов, разница между DP4ALinear и этим эталоном будет ~0.00000!
    """
    orig_shape = x.shape
    x_2d = x.reshape(-1, linear_dp4a.in_features).float()
    M = x_2d.shape[0]

    # Паддинг K (если нужен)
    if linear_dp4a.pad_k > 0:
        x_pad = F.pad(x_2d, (0, linear_dp4a.pad_k))
        w_pad = F.pad(linear_fp32.weight.detach().float(), (0, linear_dp4a.pad_k))
    else:
        x_pad = x_2d
        w_pad = linear_fp32.weight.detach().float()

    # 1. Fake-quant весов (поканальный)
    scale_w = w_pad.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
    qw = torch.round(w_pad / scale_w).clamp(-127, 127)
    w_dequant = qw * scale_w  # [out_features, K_padded]

    # 2. Fake-quant активаций (построчный)
    scale_x = x_pad.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
    qx = torch.round(x_pad / scale_x).clamp(-127, 127)
    x_dequant = qx * scale_x  # [M, K_padded]

    # 3. Чистое эталонное перемножение в PyTorch
    y_fake = x_dequant @ w_dequant.t()
    if linear_fp32.bias is not None:
        y_fake += linear_fp32.bias.detach().float()

    return y_fake.reshape(*orig_shape[:-1], linear_dp4a.out_features)

def print_detailed_metrics(name, out_test, out_ref_fp32, out_fake_quant):
    # Общая ошибка против FP32 (включает неизбежный шум квантования)
    l2_total = (torch.linalg.vector_norm(out_test - out_ref_fp32) / torch.linalg.vector_norm(out_ref_fp32).clamp_min(1e-12)).item()
    cos_sim = F.cosine_similarity(out_test.flatten(), out_ref_fp32.flatten(), dim=0).item()
    max_err = (out_test - out_ref_fp32).abs().max().item()

    # Чистая ошибка ядра против эталона Fake-Quant (должна быть ~0!)
    kernel_diff = (torch.linalg.vector_norm(out_test - out_fake_quant) / torch.linalg.vector_norm(out_fake_quant).clamp_min(1e-12)).item()

    print(f"[{name}]")
    print(f"  • Ошибка C++ ядра против Fake-Quant: {kernel_diff:.8f}  <-- (ЧИСТОТА ЯДРА: идеал = 0.0)")
    print(f"  • Общая L2-ошибка против FP32:       {l2_total:.6f} ({l2_total*100:.3f}% шум INT8)")
    print(f"  • Cosine Similarity (сходство):      {cos_sim:.6f}")
    print(f"  • Макс. абсолютная ошибка:           {max_err:.6f}")

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
iters = 20
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

out_fp32 = linear_fp32(x_fp32)
out_dp4a = linear_dp4a(x_fp32)
out_fake = compute_fake_quant_ref(linear_fp32, linear_dp4a, x_fp32)

print("\n" + "="*70)
print(f" ТЕСТ 1: КВАДРАТНЫЙ СЛОЙ ({size}x{size})")
print("="*70)
print(f"FP32 rocBLAS:  {fp32_med:.3f} ms (min: {fp32_min:.3f}, max: {fp32_max:.3f})")
print(f"FP16 rocBLAS:  {fp16_med:.3f} ms (min: {fp16_min:.3f}, max: {fp16_max:.3f})")
print(f"DP4ALinear:    {dp4a_med:.3f} ms (min: {dp4a_min:.3f}, max: {dp4a_max:.3f})")
print("-" * 70)
print(f"Ускорение относительно FP32: {fp32_med / dp4a_med:.3f}x")
print(f"Преимущество над FP16:       {fp16_med / dp4a_med:.3f}x")
print_detailed_metrics("Валидация точности 4096x4096", out_dp4a, out_fp32, out_fake)

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
    out_sdxl_ref = sdxl_l_fp32(x_sdxl)
    out_sdxl_test = sdxl_l_dp4a(x_sdxl)
    out_sdxl_fake = compute_fake_quant_ref(sdxl_l_fp32, sdxl_l_dp4a, x_sdxl)

print_detailed_metrics("Валидация SDXL 3D формы", out_sdxl_test, out_sdxl_ref, out_sdxl_fake)
print("="*70 + "\n")