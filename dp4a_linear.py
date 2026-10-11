"""Production-grade DP4ALinear without wasteful M-padding."""
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
    def __init__(self, linear: nn.Linear, ext):
        super().__init__()
        self.ext = ext
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        target_device = linear.weight.device

        self.pad_k = (64 - (self.in_features % 64)) % 64
        self.pad_n = (64 - (self.out_features % 64)) % 64

        weight = linear.weight.detach().float()
        if self.pad_n > 0 or self.pad_k > 0:
            weight = F.pad(weight, (0, self.pad_k, 0, self.pad_n))

        wq, ws = quantize_per_channel_weights(weight)
        self.register_buffer("weight_scale", ws.float().to(device=target_device))
        self.register_buffer("weight_packed", pack_i8_rows_cpu_init(wq).to(device=target_device))

        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().float().contiguous().to(device=target_device))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        if x.shape[-1] != self.in_features:
            raise ValueError(f"DP4ALinear: ожидалась размерность входа {self.in_features}, получено {x.shape[-1]}")
        if x.device != self.weight_packed.device:
            raise RuntimeError(f"DP4ALinear: вход на {x.device}, а веса слоя на {self.weight_packed.device}")

        orig_shape = x.shape
        x_dtype = x.dtype

        x_2d = x.reshape(-1, self.in_features).contiguous().float()
        M, K = x_2d.shape

        # ХОЛОСТОЙ ПАДДИНГ M ЛИКВИДИРОВАН!
        # Паддим только K по горизонтали (если K не кратно 64). Строки M НЕ трогаем!
        if self.pad_k > 0:
            x_padded = F.pad(x_2d, (0, self.pad_k, 0, 0)).contiguous()
        else:
            x_padded = x_2d

        M_real, K_padded = x_padded.shape

        # Масштаб активаций
        a_scale = float((x_padded.abs().amax() / 127.0).clamp_min(1e-12).item())

        # C++ упаковка ровно M_real строк (без 102 лишних нулей!)
        a_packed_1d = self.ext.quantize_pack_symmetric(x_padded, a_scale)
        a_packed = a_packed_1d.view(M_real, K_padded // 4)

        # Вызов ядра: передаем честный M_real! Ядро само отсечет хвостик.
        y = self.ext.fused_forward(
            a_packed, self.weight_packed, self.weight_scale,
            self.bias, a_scale,
            M_real, self.weight_packed.shape[0], K_padded // 4
        )

        # Срезаем паддинг по N (если был)
        y = y[:, :self.out_features]

        return y.reshape(*orig_shape[:-1], self.out_features).to(dtype=x_dtype)