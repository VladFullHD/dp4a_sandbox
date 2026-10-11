#include <torch/extension.h>

// Implemented in dp4a_kernels.cu (compiled as HIP by ROCm PyTorch).
torch::Tensor dp4a_mm_naive(torch::Tensor a_packed, torch::Tensor bt_packed, int64_t m, int64_t n, int64_t k4);
torch::Tensor dp4a_mm_tiled(torch::Tensor a_packed, torch::Tensor bt_packed, int64_t m, int64_t n, int64_t k4);
torch::Tensor quantize_pack_affine(torch::Tensor x, torch::Tensor scale, torch::Tensor zero_point);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mm_naive", &dp4a_mm_naive, "Naive packed INT8 GEMM via AMD sdot4");
    m.def("mm_tiled", &dp4a_mm_tiled, "Tiled packed INT8 GEMM via AMD sdot4");
    m.def("quantize_pack_affine", &quantize_pack_affine, "Affine-quantize FP32 tensor to packed INT8x4");
}
