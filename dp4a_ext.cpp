#include <torch/extension.h>

torch::Tensor dp4a_linear_fused_forward(
    torch::Tensor a_packed,
    torch::Tensor bt_packed,
    torch::Tensor weight_scales,
    c10::optional<torch::Tensor> bias,
    double a_scale,
    int64_t m, int64_t n, int64_t k4);

torch::Tensor quantize_pack_symmetric_gpu(torch::Tensor x, double scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_forward", &dp4a_linear_fused_forward, "Fused DP4A GEMM with scales and bias");
    m.def("quantize_pack_symmetric", &quantize_pack_symmetric_gpu, "Fast symmetric INT8 packing");
}