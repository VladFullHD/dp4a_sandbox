#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>
#include <cmath>

__device__ __forceinline__ int32_t sdot4(int32_t a, int32_t b, int32_t acc) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return __builtin_amdgcn_sdot4(a, b, acc, false);
#else
    return acc;
#endif
}

// FUSED TILED KERNEL: sdot4 + LDS Stride 17 + FUSED EPILOGUE (Scale + Bias + FP32 writeback)
__global__ void dp4a_mm_tiled_fused_kernel(
    const int32_t* __restrict__ a,
    const int32_t* __restrict__ bt,
    float* __restrict__ c,
    const float* __restrict__ weight_scales,
    const float* __restrict__ bias,
    float a_scale,
    int m, int n, int k4)
{
    __shared__ int32_t a_tile[128][17];
    __shared__ int32_t b_tile[64][17];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int row0 = blockIdx.y * 128 + threadIdx.y * 8;
    const int col0 = blockIdx.x * 64 + threadIdx.x * 4;

    int32_t acc[8][4] = {0};

    const int4* __restrict__ a_vec = reinterpret_cast<const int4*>(a);
    const int4* __restrict__ bt_vec = reinterpret_cast<const int4*>(bt);
    const int k_vec = k4 / 4;

    const int row_a0 = tid / 4;
    const int col_v_a = tid % 4;
    const int row_a1 = row_a0 + 64;
    const int ar0 = blockIdx.y * 128 + row_a0;
    const int ar1 = blockIdx.y * 128 + row_a1;

    const int row_b = tid / 4;
    const int col_v_b = tid % 4;
    const int br = blockIdx.x * 64 + row_b;

    for (int k0 = 0; k0 < k4; k0 += 16) {
        const int k_vec_base = k0 / 4 + col_v_a;

        int4 va0 = (ar0 < m && k_vec_base < k_vec) ? a_vec[static_cast<int64_t>(ar0) * k_vec + k_vec_base] : make_int4(0, 0, 0, 0);
        int4 va1 = (ar1 < m && k_vec_base < k_vec) ? a_vec[static_cast<int64_t>(ar1) * k_vec + k_vec_base] : make_int4(0, 0, 0, 0);

        const int a_col = col_v_a * 4;
        a_tile[row_a0][a_col + 0] = va0.x; a_tile[row_a0][a_col + 1] = va0.y;
        a_tile[row_a0][a_col + 2] = va0.z; a_tile[row_a0][a_col + 3] = va0.w;
        a_tile[row_a1][a_col + 0] = va1.x; a_tile[row_a1][a_col + 1] = va1.y;
        a_tile[row_a1][a_col + 2] = va1.z; a_tile[row_a1][a_col + 3] = va1.w;

        const int k_vec_base_b = k0 / 4 + col_v_b;
        int4 vb = (br < n && k_vec_base_b < k_vec) ? bt_vec[static_cast<int64_t>(br) * k_vec + k_vec_base_b] : make_int4(0, 0, 0, 0);

        const int b_col = col_v_b * 4;
        b_tile[row_b][b_col + 0] = vb.x; b_tile[row_b][b_col + 1] = vb.y;
        b_tile[row_b][b_col + 2] = vb.z; b_tile[row_b][b_col + 3] = vb.w;

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            int32_t a_vals[8];
            #pragma unroll
            for (int r = 0; r < 8; ++r) a_vals[r] = a_tile[threadIdx.y * 8 + r][kk];

            int32_t b_vals[4];
            #pragma unroll
            for (int ci = 0; ci < 4; ++ci) b_vals[ci] = b_tile[threadIdx.x * 4 + ci][kk];

            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                #pragma unroll
                for (int ci = 0; ci < 4; ++ci) {
                    acc[r][ci] = sdot4(a_vals[r], b_vals[ci], acc[r][ci]);
                }
            }
        }
        __syncthreads();
    }

    // FUSED EPILOGUE: Прямая запись во float32 с поканальным масштабом и bias
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        #pragma unroll
        for (int ci = 0; ci < 4; ++ci) {
            const int r_glob = row0 + r;
            const int c_glob = col0 + ci;

            if (r_glob < m && c_glob < n) {
                float w_scale = weight_scales[c_glob];
                float val = static_cast<float>(acc[r][ci]) * (a_scale * w_scale);
                if (bias != nullptr) {
                    val += bias[c_glob];
                }
                c[static_cast<int64_t>(r_glob) * n + c_glob] = val;
            }
        }
    }
}

// Симметричный C++ квантователь активаций
__global__ void symmetric_quantize_pack_kernel(
    const float* __restrict__ x,
    float scale,
    int32_t* __restrict__ packed,
    int64_t count) {
    const int64_t group = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t base = group * 4;
    if (base >= count) return;
    const float inv_scale = 1.0f / fmaxf(scale, 1.0e-12f);
    uint32_t word = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int64_t i = base + j;
        int32_t q = 0;
        if (i < count) {
            q = static_cast<int32_t>(nearbyintf(x[i] * inv_scale));
            q = q < -127 ? -127 : (q > 127 ? 127 : q);
        }
        word |= (static_cast<uint32_t>(q) & 0xffu) << (8 * j);
    }
    packed[group] = static_cast<int32_t>(word);
}

// Обёртка Fused-слоя с проверками совпадения устройств
torch::Tensor dp4a_linear_fused_forward(
    torch::Tensor a_packed,
    torch::Tensor bt_packed,
    torch::Tensor weight_scales,
    c10::optional<torch::Tensor> bias,
    double a_scale,
    int64_t m, int64_t n, int64_t k4) 
{
    TORCH_CHECK(a_packed.is_cuda() && bt_packed.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(a_packed.device() == bt_packed.device(), "a_packed and bt_packed must be on the same device");
    TORCH_CHECK(weight_scales.device() == a_packed.device(), "weight_scales must be on the same device as inputs");
    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->device() == a_packed.device(), "bias must be on the same device as inputs");
    }

    c10::cuda::CUDAGuard device_guard(a_packed.device());
    auto stream = c10::cuda::getCurrentCUDAStream(a_packed.get_device());

    auto out = torch::empty({m, n}, a_packed.options().dtype(torch::kFloat32));

    const float* bias_ptr = bias.has_value() && bias->defined() ? bias->data_ptr<float>() : nullptr;

    dim3 block(16, 16);
    dim3 grid((n + 63) / 64, (m + 127) / 128);

    dp4a_mm_tiled_fused_kernel<<<grid, block, 0, stream>>>(
        a_packed.data_ptr<int32_t>(),
        bt_packed.data_ptr<int32_t>(),
        out.data_ptr<float>(),
        weight_scales.data_ptr<float>(),
        bias_ptr,
        static_cast<float>(a_scale),
        static_cast<int>(m), static_cast<int>(n), static_cast<int>(k4)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// Быстрая симметричная упаковка C++
torch::Tensor quantize_pack_symmetric_gpu(torch::Tensor x, double scale) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "x must be contiguous GPU tensor");
    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());

    auto out = torch::empty({x.numel() / 4}, x.options().dtype(torch::kInt32));
    constexpr int threads = 256;
    const int64_t groups = x.numel() / 4;
    const int blocks = static_cast<int>((groups + threads - 1) / threads);

    symmetric_quantize_pack_kernel<<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(), static_cast<float>(scale), out.data_ptr<int32_t>(), x.numel()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}