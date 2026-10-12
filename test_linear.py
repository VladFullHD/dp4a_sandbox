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
    orig_shape = x.shape
    x_2d = x.reshape(-1, linear_dp4a.in_features).float()
    if linear_dp4a.pad_k > 0:
        x_pad = F.pad(x_2d, (0, linear_dp4a.pad_k))
        w_pad = F.pad(linear_fp32.weight.detach().float(), (0, linear_dp4a.pad_k))
    else:
        x_pad = x_2d
        w_pad = linear_fp32.weight.detach().float()

    scale_w = w_pad.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
    qw = torch.round(w_pad / scale_w).clamp(-127, 127)
    w_dequant = qw * scale_w

    scale_x = x_pad.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
    qx = torch.round(x_pad / scale_x).clamp(-127, 127)
    x_dequant = qx * scale_x

    y_fake = x_dequant @ w_dequant.t()
    if linear_fp32.bias is not None:
        y_fake += linear_fp32.bias.detach().float()

    return y_fake.reshape(*orig_shape[:-1], linear_dp4a.out_features)

def print_detailed_metrics(name, out_test, out_ref_fp32, out_fake_quant):
    l2_total = (torch.linalg.vector_norm(out_test - out_ref_fp32) / torch.linalg.vector_norm(out_ref_fp32).clamp_min(1e-12)).item()
    cos_sim = F.cosine_similarity(out_test.flatten(), out_ref_fp32.flatten(), dim=0).item()
    max_err = (out_test - out_ref_fp32).abs().max().item()
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
# 1. ТЕСТ 4096x4096 (Синхронизированные веса и Fake-Quant)
# -----------------------------------------------------------------
size = 4096
iters = 20
linear_fp32 = nn.Linear(size, size, bias=True).cuda()
linear_fp16 = nn.Linear(size, size, bias=True).half().cuda()
linear_fp16.weight.data.copy_(linear_fp32.weight.data.half())
linear_fp16.bias.data.copy_(linear_fp32.bias.data.half())

linear_dp4a = DP4ALinear(linear_fp32, ext)

x_fp32 = torc