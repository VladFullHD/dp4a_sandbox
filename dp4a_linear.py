"""Production-grade DP4ALinear module for AMD RDNA 2 (gfx1030)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def quantize_per_channel_weights(w: torch.Tensor):
    scales = w.detach().abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
    q = torch.round(w / scales).clamp(-127, 127).to(torch.int8)
    return q, scales.squeeze(1)


def pack_i8_rows_cpu_init(q: torch.Tensor) -> torch.Tensor:
    rows, k = q.shape
    v = q.contiguous().to(torch.int32).reshape(rows, k // 4, 4)
    b0 = v[..., 0] & 255
    b1 = (v[..., 1] & 255) << 8
    b2 = (v[..., 2] & 255) << 16
    b3 = (v[..., 3] & 255) << 24
    return (b0 | b1 | b2 | b3).contiguous()


class DP4ALinear(nn.Module):
    def __init__(self, linear: nn.Linear, ext, kernel: str = "tiled"):
        super().__init__()
        self.ext = ext
        self.kernel = kernel
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        # Паддинг для весов (K и N кратны 64)
        self.pad_k = (64 - (self.in_features % 64)) % 64
        self.pad_n = (64 - (self.out_features % 64)) % 64

        weight = linear.weight.detach().float()
        if self.pad_n > 0 or self.pad_k > 0:
            weight = F.pad(weight, (0, self.pad_k, 0, self.pad_n))

        wq, ws = quantize_per_channel_weights(weight)
        self.register_buffer("weight_scale", ws.float().cuda())
        self.register_buffer("weight_packed", pack_i8_rows_cpu_init(wq).cuda())

        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().float().contiguous())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_dtype = x.dtype

        # 1. Разворачиваем любые N-мерные тензоры в 2D [M, K]
        x_2d = x.reshape(-1, self.in_features).contiguous().float()
        M, K = x_2d.shape

        # 2. Динамический паддинг входов: M к 128, K к 64
        pad_m = (128 - (M % 128)) % 128
        pad_k = self.pad_k

        if pad_m > 0 or pad_k > 0:
            x_padded = F.pad(x_2d, (0, pad_k, 0, pad_m)).contiguous()
        else:
            x_padded = x_2d

        M_padded, K_padded = x_padded.shape

        # 3. Быстрая C++ упаковка активаций (~0.3 мс)
        a_scale = (x_padded.abs().amax().clamp_min(1e-12) / 127.0).reshape(())
        zero_point = torch.tensor(0, dtype=torch.int32, device=x.device)
        
        a_packed_1d = self.ext.quantize_pack_affine(x_padded, a_scale, zero_point)
        a_packed = a_packed_1d.view(M_padded, K_padded // 4)

        # 4. Аппаратный DP4A GEMM
        mm = self.ext.mm_tiled if self.kernel == "tiled" else self.ext.mm_naive
        acc = mm(a_packed, self.weight_packed, M_padded, self.weight_packed.shape[0], K_padded // 4)

        # 5. Поканальное деквантование и bias
        y = acc.float() * (a_scale * self.weight_scale.unsqueeze(0))

        if self.bias is not None:
            y = y[:, :self.out_features] + self.bias
        else:
            y = y[:, :self.out_features]

        # 6. Срезаем паддинг по M
        if pad_m > 0:
            y = y[:M, :]

        return y.reshape(*orig_shape[:-1], self.out_features).to(dtype=x_dtype)