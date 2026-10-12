#include <torch/extension.h>

// Fused API
torch::Tensor dp4a_linear_fused_forward(
    torch::Tensor a_packed,
    torch::Tensor bt_packed,
    torch::Tensor weight_scales,
    c10::optional<torch::Tensor> bias,
    torch::Tensor a_scale_tensor,
    int64_t m, int64_t n, int64_t k4);

torch::Tensor quantize_pack_symmetric_gpu(
    torch::Tensor x,
    torch::Tensor scale_tensor);

// Legacy API (для совместимости со старым benchmark.py)
torch::Tensor dp4a_mm_naive(torch::Tensor a, torch::Tensor bt, int64_t m, int64_t n, int64_t k4);
torch::Tensor dp4a_mm_tiled(torch::Tensor a, torch::Tensor bt, int64_t m, int64_t n, int64_t k4);
torch::Tensor quantize_pack_affine(torch::Tensor x, torch::Tensor scale, torch::Tensor zero_point);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Новые оптимизированные вызовы
    m.def("fused_forward", &dp4a_linear_fused_forward, "Fused DP4A GEMM/GEMV with scale tensor and bias");
    m.def("quantize_pack_symmetric", &quantize_pack_symmetric_gpu, "Zero-sync symmetric INT8 packing");

    // Обратная совместимость для benchmark.py
    m.def("mm_naive", &dp4a_mm_naive, "Legacy naive GEMM");
    m.def("mm_tiled", &dp4a_mm_tiled, "Legacy tiled GEMM");
    m.def("quantize_pack_affine", &quantize_pack_affine, "Legacy affine packing");
}