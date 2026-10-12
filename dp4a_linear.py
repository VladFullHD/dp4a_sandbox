"""Production-grade DP4ALinear with native FP16/BF16/FP32 zero-copy input packing."""
import copy
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
        self._ext_ref = ext
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        target_device = linear.weight.device

        self.pad_k = (64 - (self.in_features % 64)) % 64

        weight = linear.weight.detach().float()
        if self.pad_k > 0:
            weight = F.pad(weight, (0, self.pad_k))

        wq, ws = quantize_per_channel_weights(weight)
        self.register_buffer("weight_scale", ws.float().to(device=target_device))
        self.register_buffer("weight_packed", pack_i8_rows_cpu_init(wq).to(device=target_device))

        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().float().contiguous().to(device=target_device))
        else:
            self.bias = None

    def _apply(self, fn):
        """Защита от model.half(): weight_scale всегда обязан оставаться float32!"""
        super()._apply(fn)
        if hasattr(self, "weight_scale") and self.weight_scale is not None:
            self.weight_scale = self.weight_scale.to(torch.float32)
        return self

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "_ext_ref":
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 1:
            raise ValueError("DP4ALinear expects an input with at least one dimension")
        if x.shape[-1] != self.in_features:
            raise ValueError(f"DP4ALinear: expected in_features={self.in_features}, got {x.shape[-1]}")
        if not x.is_floating_point():
            raise TypeError("DP4ALinear only supports floating point inputs")
        if x.numel() == 0:
            return x.new_empty((*x.shape[:-1], self.out_features))
        if x.device != self.weight_packed.device:
            raise RuntimeError(f"DP4ALinear: input on {x.device}, layer on {self.weight_packed.device}")

        orig_shape = x.shape
        x_dtype = x.dtype

        # НИКАКОГО .float()! Сохраняем исходный dtype (float16/bfloat16/float32) без копирования!
        x_2d = x.reshape(-1, self.in_features).contiguous()
        M, K = x_2d.shape

        if self.pad_k > 0:
            x_padded = F.pad(x_2d, (0, self.pad_k, 0, 0)).contiguous()
        else:
            x_padded = x_2d

        M_real, K_padded = x_padded.shape

        # Построчный масштаб (считается во float32 для точности, но сам x_padded НЕ конвертируется)
        a_scale_per_row = (x_padded.abs().amax(dim=1).float() / 127.0).clamp_min(1e-12).contiguous()

        # Быстрая C++ упаковка читает x_padded напрямую в его нативном типе (half/float)
        a_packed_1d = self._ext_ref.quantize_pack_symmetric(x_padded, a_scale_per_row, K_padded)
        a_packed = a_packed_1d.view(M_real, K_padded // 4)

        # Fused GEMM
        y = self._ext_ref.fused_forward(
            a_packed, self.weight_packed, self.weight_scale,
            self.bias, a_scale_per_row,
            M_real, self.out_features, K_padded // 4
        )

        return y.reshape(*orig_shape[:-1], self.out_features).to(dtype=x_dtype)