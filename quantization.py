"""Stage 3: affine activation quantization + int8x4 packing with zero-encompassing range."""
import torch


def quantize_pack_affine(x: torch.Tensor, ext):
    if not x.is_cuda:
        raise ValueError("x must be on the GPU")
    if x.numel() == 0:
        return torch.empty(0, dtype=torch.int32, device=x.device), torch.tensor(1.0, device=x.device), torch.tensor(0, device=x.device), torch.tensor(0.0, device=x.device), torch.tensor(0.0, device=x.device)
    if x.numel() % 4 != 0:
        raise ValueError("tensor element count must be divisible by 4")

    x = x.contiguous().float()

    # ФИКС БАГА СДВИГА НУЛЯ: диапазон квантования обязан включать 0,
    # иначе чисто положительные или чисто отрицательные входы насыщают zero_point на -128!
    lo = torch.minimum(torch.amin(x), torch.zeros((), device=x.device, dtype=torch.float32))
    hi = torch.maximum(torch.amax(x), torch.zeros((), device=x.device, dtype=torch.float32))

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