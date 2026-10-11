"""Stage 3: affine activation quantization + int8x4 packing prototype.

Min/max reduction currently uses PyTorch's optimized GPU reduction; the custom HIP
kernel then quantizes and packs four signed int8 values per int32 word.
"""
from __future__ import annotations
import torch


def quantize_pack_affine(x: torch.Tensor, ext):
    if not x.is_cuda:
        raise ValueError("x must be on the GPU")
    x = x.contiguous().float()
    if x.numel() % 4:
        raise ValueError("tensor element count must be divisible by 4")
    lo = torch.amin(x)
    hi = torch.amax(x)
    scale = ((hi - lo) / 255.0).clamp_min(1.0e-12).reshape(())
    zp = torch.round(-128.0 - lo / scale).clamp(-128, 127).to(torch.int32).reshape(())
    packed = ext.quantize_pack_affine(x, scale, zp)
    return packed, scale, zp, lo, hi


def unpack_int8x4(packed: torch.Tensor, count: int) -> torch.Tensor:
    """Debug helper: unpack packed signed bytes to int8 vector."""
    words = packed.to(torch.int64).reshape(-1, 1)
    shifts = torch.tensor([0, 8, 16, 24], device=packed.device, dtype=torch.int64)
    vals = ((words >> shifts) & 0xff).to(torch.uint8).view(torch.int8).reshape(-1)
    return vals[:count]
