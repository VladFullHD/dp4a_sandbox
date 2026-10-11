#!/usr/bin/env python3
"""ROCm/RDNA2 DP4A microbenchmark.

Runs FP32 and FP16 torch.matmul references, naive and LDS-tiled INT8x4 DP4A, and
an affine quantize+pack microbenchmark. Tiled DP4A is tested on the requested size.
For desktop responsiveness, naive DP4A uses --naive-size (default 512) rather than
running the very serial 4096^3 kernel on the display GPU; pass --naive-size 4096 to
measure it at full size after the tiled kernel has been validated.
"""
from __future__ import annotations
import argparse
import os
import time
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parent


def load_ext():
    build_dir = ROOT / "build"
    build_dir.mkdir(exist_ok=True)
    os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx1030")
    os.environ.setdefault("MAX_JOBS", "2")

    # ФИКС БАГА FEDORA / GCC: убираем флаг -isystem /usr/include, ломающий stdlib.h
    import torch.utils.cpp_extension as cpp_ext
    orig_include_paths = cpp_ext.include_paths
    def patched_include_paths(*args, **kwargs):
        return [p for p in orig_include_paths(*args, **kwargs) if os.path.normpath(p) != "/usr/include"]
    cpp_ext.include_paths = patched_include_paths

    return load(
        name="dp4a_sandbox_ext",
        sources=[str(ROOT / "dp4a_ext.cpp"), str(ROOT / "dp4a_kernels.cu")],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        build_directory=str(build_dir),
        verbose=True,
        with_cuda=True,
    )


def sync():
    torch.cuda.synchronize()


def bench(fn, warmup=2, iters=5):
    for _ in range(warmup):
        y = fn()
        del y
    sync()
    samples = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        y = fn()
        sync()
        samples.append((time.perf_counter() - t0) * 1000.0)
        del y
    return sorted(samples)[len(samples) // 2], samples


def pack_i8_rows(q: torch.Tensor) -> torch.Tensor:
    rows, k = q.shape
    if k % 4:
        raise ValueError("K must be divisible by 4")
    v = q.contiguous().to(torch.int32).view(rows, k // 4, 4)
    return ((v[..., 0] & 255) | ((v[..., 1] & 255) << 8) |
            ((v[..., 2] & 255) << 16) | ((v[..., 3] & 255) << 24)).contiguous()


def symmetric_quantize(x: torch.Tensor):
    scale = x.abs().amax().clamp_min(1e-12) / 127.0
    q = torch.round(x / scale).clamp(-127, 127).to(torch.int8)
    return q, scale


def rel_error(y, ref):
    yf, rf = y.float(), ref.float()
    return (torch.linalg.vector_norm(yf - rf) / torch.linalg.vector_norm(rf).clamp_min(1e-12)).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--naive-size", type=int, default=512,
                        help="size for serial naive DP4A; use 4096 to benchmark at full size")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--skip-affine-pack", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("PyTorch did not detect a GPU. Run inside the ROCm ComfyUI venv with HSA_OVERRIDE_GFX_VERSION=10.3.0.")
    n = args.size
    if n % 32 or n % 4:
        raise SystemExit("--size must be divisible by 32 for the tiled kernel")
    if args.naive_size % 4:
        raise SystemExit("--naive-size must be divisible by 4")
    print(f"torch={torch.__version__}; hip={getattr(torch.version, 'hip', None)}; device={torch.cuda.get_device_name(0)}")
    print(f"sizes: baseline/tiled={n}x{n}, naive={args.naive_size}x{args.naive_size}")
    ext = load_ext()
    torch.manual_seed(7)

    # First do correctness on a small matrix, including exact equality between the
    # naive and tiled integer accumulators (same quantized inputs).
    s = 128
    a0 = torch.randn((s, s), device="cuda", dtype=torch.float32)
    b0 = torch.randn((s, s), device="cuda", dtype=torch.float32)
    aq0, sa0 = symmetric_quantize(a0)
    bq0, sb0 = symmetric_quantize(b0.t().contiguous())
    ap0, bp0 = pack_i8_rows(aq0), pack_i8_rows(bq0)
    # Use CPU int64 as an independent exact reference; int32 GEMM support varies by backend.
    ref0 = (aq0.cpu().to(torch.int64) @ bq0.cpu().to(torch.int64).t()).to(device="cuda", dtype=torch.int32)
    got_naive = ext.mm_naive(ap0, bp0, s, s, s // 4)
    got_tiled = ext.mm_tiled(ap0, bp0, s, s, s // 4)
    sync()
    if not torch.equal(got_naive, ref0):
        raise RuntimeError(f"naive DP4A incorrect; max abs int32 diff={(got_naive-ref0).abs().max().item()}")
    if not torch.equal(got_tiled, ref0):
        raise RuntimeError(f"tiled DP4A incorrect; max abs int32 diff={(got_tiled-ref0).abs().max().item()}")
    print("PASS: naive and tiled DP4A accumulators match int32 reference exactly on 128x128.")

    a = torch.randn((n, n), device="cuda", dtype=torch.float32)
    b = torch.randn((n, n), device="cuda", dtype=torch.float32)
    q_a, scale_a = symmetric_quantize(a)
    q_bt, scale_b = symmetric_quantize(b.t().contiguous())
    a_pack = pack_i8_rows(q_a)
    bt_pack = pack_i8_rows(q_bt)

    print("\nTiming FP32 torch.matmul...")
    fp32_ms, _ = bench(lambda: a @ b, iters=args.iters)
    print(f"torch.matmul FP32: {fp32_ms:.3f} ms")
    a16, b16 = a.half(), b.half()
    print("Timing FP16 torch.matmul...")
    fp16_ms, _ = bench(lambda: a16 @ b16, iters=args.iters)
    print(f"torch.matmul FP16: {fp16_ms:.3f} ms")

    print("\nTiming tiled DP4A at full size...")
    tiled_ms, _ = bench(lambda: ext.mm_tiled(a_pack, bt_pack, n, n, n // 4), warmup=1, iters=args.iters)
    acc = ext.mm_tiled(a_pack, bt_pack, n, n, n // 4)
    dp4a_y = acc.float() * scale_a * scale_b
    fp32_ref = a @ b
    sync()
    print(f"DP4A tiled kernel-only: {tiled_ms:.3f} ms")
    print(f"DP4A dequantized relative L2 error vs FP32: {rel_error(dp4a_y, fp32_ref):.6f}")
    print(f"kernel-only speedup vs FP32: {fp32_ms / tiled_ms:.3f}x")
    print("Note: this is kernel-only; it excludes activation quantization and packing.")

    ns = args.naive_size
    if ns == n:
        an, bn = a, b
        anp, bnp = a_pack, bt_pack
        san, sbn = scale_a, scale_b
    else:
        an = torch.randn((ns, ns), device="cuda", dtype=torch.float32)
        bn = torch.randn((ns, ns), device="cuda", dtype=torch.float32)
        qan, san = symmetric_quantize(an)
        qbn, sbn = symmetric_quantize(bn.t().contiguous())
        anp, bnp = pack_i8_rows(qan), pack_i8_rows(qbn)
    print(f"\nTiming naive DP4A at {ns}x{ns}...")
    naive_ms, _ = bench(lambda: ext.mm_naive(anp, bnp, ns, ns, ns // 4), warmup=1, iters=args.iters)
    print(f"DP4A naive kernel-only: {naive_ms:.3f} ms")

    if not args.skip_affine_pack:
        from quantization import quantize_pack_affine, unpack_int8x4
        x = torch.randn((n, n), device="cuda", dtype=torch.float32).contiguous()
        lo, hi = torch.amin(x), torch.amax(x)
        sc = ((hi - lo) / 255.0).clamp_min(1e-12).reshape(())
        zp = torch.round(-128.0 - lo / sc).clamp(-128, 127).to(torch.int32).reshape(())
        def affine_pack_call():
            lo_ = torch.amin(x)
            hi_ = torch.amax(x)
            sc_ = ((hi_ - lo_) / 255.0).clamp_min(1e-12).reshape(())
            zp_ = torch.round(-128.0 - lo_ / sc_).clamp(-128, 127).to(torch.int32).reshape(())
            return ext.quantize_pack_affine(x, sc_, zp_)
        affine_ms, _ = bench(affine_pack_call, warmup=1, iters=args.iters)
        packed, sc, zp, lo, hi = quantize_pack_affine(x, ext)
        unpacked = unpack_int8x4(packed, x.numel()).float()
        xhat = (unpacked - zp.float()) * sc
        print(f"Affine min/max + custom quantize/pack: {affine_ms:.3f} ms")
        print(f"Affine pack validation: count={unpacked.numel()}, scale={sc.item():.7g}, zero_point={zp.item()}, min={lo.item():.6g}, max={hi.item():.6g}, rel L2 reconstruction error={rel_error(xhat.view_as(x), x):.6f}")

    print("\nDone. Compare end-to-end timings, not just kernel-only numbers, before making performance claims.")


if __name__ == "__main__":
    main()
