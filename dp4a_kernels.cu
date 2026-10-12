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

// 1. СПЕЦИАЛИЗИРОВАННОЕ GEMV ЯДРО ДЛЯ M = 1
__global__ void dp4a_gemv_fused_kernel(
    const int32_t* __restrict__ a,
    const int32_t* __restrict__ bt,
    float* __restrict__ c,
    const float* __restrict__ weight_scales,
    const float* __restrict__ bias,
    const float* __restrict__ a_scale_ptr,
    int n, int k4)
{
    const int tid = threadIdx.x;
    const int col0 = blockIdx.x * 4;

    int32_t sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;

    for (int p = tid; p < k4; p += 256) {
        int32_t a_val = a[p];
        if (col0 + 0 < n) sum0 = sdot4(a_val, bt[static_cast<int64_t>(col0 + 0) * k4 + p], sum0);
        if (col0 + 1 < n) sum1 = sdot4(a_val, bt[static_cast<int64_t>(col0 + 1) * k4 + p], sum1);
        if (col0 + 2 < n) sum2 = sdot4(a_val, bt[static_cast<int64_t>(col0 + 2) * k4 + p], sum2);
        if (col0 + 3 < n) sum3 = sdot4(a_val, bt[static_cast<int64_t>(col0 + 3) * k4 + p], sum3);
    }

    __shared__ int32_t s_red[4][256];
    s_red[0][tid] = sum0;
    s_red[1][tid] = sum1;
    s_red[2][tid] = sum2;
    s_red[3][tid] = sum3;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            s_red[0][tid] += s_red[0][tid + s];
            s_red[1][tid] += s_red[1][tid + s];
            s_red[2][tid] += s_red[2][tid + s];
            s_red[3][tid] += s_red[3][tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float a_scale = *a_scale_ptr;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int col = col0 + i;
            if (col < n) {
                float val = static_cast<float>(s_red[i][0]) * (a_scale * weight_scales[col]);
                if (bias != nullptr) val += bias[col];
                c[col] = val;
            }
        }
    }
}

// 2. FUSED GEMM ЯДРО ДЛЯ M > 1
__global__ void dp4a_mm_tiled_fused_kernel(
    const int32_t* __restrict__ a,
    const int32_t* __restrict__ bt,
    float* __restrict__ c,
    const float* __restrict__ weight_scales,
    const float* __restrict__ bias,
    const float* __restrict__ a_scale_ptr,
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
            for (int ci = 0; ci < 4; ++ci) {
                #pragma unroll
                for (int r = 0; r < 8; ++r) {
                    acc[r][ci] = sdot4(a_vals[r], b_vals[ci], acc[r][ci]);
                }
            }
        }
        __syncthreads();
    }

    float a_scale = *a_scale_ptr;

    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        #pragma unroll
        for (int ci = 0; ci < 4; ++ci) {
            const int r_glob = row0 + r;
            const int c_glob = col0 + ci;

            if (r_glob < m && c_glob < n) {
                float w_scale = weight_scales[c_glob];
                float val = static_cast<float>(acc[r][ci]) * (a_scale * w_scale);
                if (bias != nullptr) val += bias[c_glob];
                c[static_cast<int64_t>(r_glob) * n + c_glob] = val;
            }
        }
    }
}

// 3. ЧЕСТНОЕ INT32 TILED-ЯДРО (ДЛЯ ТОЧНОЙ ВЕРИФИКАЦИИ В BENCHMARK.PY)
__global__ void dp4a_mm_tiled_int32_kernel(
    const int32_t* __restrict__ a,
    const int32_t* __restrict__ bt,
    int32_t* __restrict__ c,
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
            for (int ci = 0; ci < 4; ++ci) {
                #pragma unroll
                for (int r = 0; r < 8; ++r) {
                    acc[r][ci] = sdot4(a_vals[r], b_vals[ci], acc[r][ci]);
                }
            }
        }
        __syncthreads();
    }

    // ЧЕСТНАЯ ЗАПИСЬ INT32 БЕЗ ПРОМЕЖУТОЧНОГО FLOAT32
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        #pragma unroll
        for (int ci = 0; ci < 4; ++ci) {
            const int r_glob = row0 + r;
            const int c_glob = col0 + ci;
            if (r_glob < m && c_glob < n) {
                c[static_cast<int64_t>(r_glob) * n + c_glob] = acc[r][ci];
            }
        }
    }
}

// 4. СИММЕТРИЧНЫЙ КВАНТОВАТЕЛЬ
__global__ void symmetric_quantize_pack_kernel(
    const float* __restrict__ x,
    const float* __restrict__ scale_ptr,
    int32_t* __restrict__ packed,
    int64_t count) {
    const int64_t group = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t base = group * 4;
    if (base >= count) return;
    const float inv_scale = 1.0f / fmaxf(*scale_ptr, 1.0e-12f);
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

// =========================================================================
// ОБЁРТКИ C++ С ПОЛНОЙ ВАЛИДАЦИЕЙ АРГУМЕНТОВ (TORCH_CHECK)
// =========================================================================

torch::Tensor dp4a_linear_fused_forward(
    torch::Tensor a_packed,
    torch::Tensor bt_packed,
    torch::Tensor weight_scales,
    c10::optional<torch::Tensor> bias,
    torch::Tensor a_scale_tensor,
    int64_t m, int64_t n, int64_t k4) 
{
    TORCH_CHECK(a_packed.is_cuda() && bt_packed.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(a_packed.scalar_type() == torch::kInt32 && bt_packed.scalar_type() == torch::kInt32, "Packed inputs must be int32");
    TORCH_CHECK(weight_scales.is_cuda() && weight_scales.scalar_type() == torch::kFloat32, "weight_scales must be float32 CUDA tensor");
    TORCH_CHECK(a_scale_tensor.is_cuda() && a_scale_tensor.scalar_type() == torch::kFloat32 && a_scale_tensor.numel() == 1, "a_scale must be scalar float32 CUDA tensor");
    TORCH_CHECK(a_packed.is_contiguous() && bt_packed.is_contiguous() && weight_scales.is_contiguous(), "Inputs must be contiguous");
    TORCH_CHECK(a_packed.device() == bt_packed.device() && a_packed.device() == weight_scales.device(), "Device mismatch");
    TORCH_CHECK(a_packed.size(0) == m && a_packed.size(1) == k4, "A packed shape mismatch");
    TORCH_CHECK(bt_packed.size(0) == n && bt_packed.size(1) == k4, "B^T packed shape mismatch");
    TORCH_CHECK(weight_scales.numel() >= n, "weight_scales size must be >= n");
    TORCH_CHECK(m > 0 && n > 0 && k4 > 0, "Dimensions must be positive");
    TORCH_CHECK(k4 % 16 == 0, "k4 must be divisible by 16");

    const float* bias_ptr = nullptr;
    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->is_cuda() && bias->device() == a_packed.device(), "bias device mismatch");
        TORCH_CHECK(bias->scalar_type() == torch::kFloat32 && bias->is_contiguous(), "bias must be contiguous float32");
        TORCH_CHECK(bias->numel() >= n, "bias size must be >= n (padded width)");
        bias_ptr = bias->data_ptr<float>();
    }

    c10::cuda::CUDAGuard device_guard(a_packed.device());
    auto stream = c10::cuda::getCurrentCUDAStream(a_packed.get_device());
    auto out = torch::empty({m, n}, a_packed.options().dtype(torch::kFloat32));

    if (m == 1) {
        dim3 block(256);
        dim3 grid((n + 3) / 4);
        dp4a_gemv_fused_kernel<<<grid, block, 0, stream>>>(
            a_packed.data_ptr<int32_t>(),
            bt_packed.data_ptr<int32_t>(),
            out.data_ptr<float>(),
            weight_scales.data_ptr<float>(),
            bias_ptr,
            a_scale_tensor.data_ptr<float>(),
            static_cast<int>(n), static_cast<int>(k4)
        );
    } else {
        dim3 block(16, 16);
        dim3 grid((n + 63) / 64, (m + 127) / 128);
        dp4a_mm_tiled_fused_kernel<<<grid, block, 0, stream>>>(
            a_packed.data_ptr<int32_t>(),
            bt_packed.data_ptr<int32_t>(),
            out.data_ptr<float>(),
            weight_scales.data_ptr<float>(),
            bias_ptr,
            a_scale_tensor.data_ptr<float>(),
            static_cast<int>(m), static_cast<int>(n), static_cast<int>(k4)
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor quantize_pack_symmetric_gpu(torch::Tensor x, torch::Tensor scale_tensor) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "x must be contiguous GPU tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(scale_tensor.is_cuda() && scale_tensor.scalar_type() == torch::kFloat32 && scale_tensor.numel() == 1, "scale must be scalar CUDA float32");
    if (x.numel() == 0) return torch::empty({0}, x.options().dtype(torch::kInt32));
    TORCH_CHECK(x.numel() % 4 == 0, "x.numel() must be divisible by 4");

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
    auto out = torch::empty({x.numel() / 4}, x.options().dtype(torch::kInt32));

    constexpr int threads = 256;
    const int64_t groups = x.numel() / 4;
    const int blocks = static_cast<int>((groups + threads - 1) / threads);

    symmetric_quantize_pack_kernel<<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(), scale_tensor.data_ptr<float>(), out.data_ptr<int32_t>(), x.numel()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// 5. LEGACY ФУНКЦИИ
__global__ void dp4a_mm_naive_kernel(
    const int32_t* __restrict__ a,
    const int32_t* __restrict__ bt,
    int32_t* __restrict__ c,
    int m, int n, int k4) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= n) return;
    int32_t acc = 0;
    const int32_t* arow = a + static_cast<int64_t>(row) * k4;
    const int32_t* brow = bt + static_cast<int64_t>(col) * k4;
    for (int p = 0; p < k4; ++p) acc = sdot4(arow[p], brow[p], acc);
    c[static_cast<int64_t>(row) * n + col] = acc;
}

__global__ void affine_quantize_pack_kernel(
    const float* __restrict__ x,
    const float* __restrict__ scale_ptr,
    const int32_t* __restrict__ zp_ptr,
    int32_t* __restrict__ packed,
    int64_t count) {
    const int64_t group = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t base = group * 4;
    if (base >= count) return;
    const float scale = fmaxf(*scale_ptr, 1.0e-12f);
    const int32_t zp = *zp_ptr;
    uint32_t word = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int64_t i = base + j;
        int32_t q = 0;
        if (i < count) {
            q = static_cast<int32_t>(nearbyintf(x[i] / scale)) + zp;
            q = q < -128 ? -128 : (q > 127 ? 127 : q);
        }
        word |= (static_cast<uint32_t>(q) & 0xffu) << (8 * j);
    }
    packed[group] = static_cast<int32_t>(word);
}

torch::Tensor dp4a_mm_naive(torch::Tensor a, torch::Tensor bt, int64_t m, int64_t n, int64_t k4) {
    c10::cuda::CUDAGuard device_guard(a.device());
    auto stream = c10::cuda::getCurrentCUDAStream(a.get_device());
    auto out = torch::empty({m, n}, a.options().dtype(torch::kInt32));
    dim3 block(16, 16);
    dim3 grid((n + 15) / 16, (m + 15) / 16);
    dp4a_mm_naive_kernel<<<grid, block, 0, stream>>>(a.data_ptr<int32_t>(), bt.data_ptr<int32_t>(), out.data_ptr<int32_t>(), static_cast<int>(m), static_cast<int>(n), static_cast<int>(k4));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor dp4a_mm_tiled(torch::Tensor a, torch::Tensor bt, int64_t m, int64_t n, int64_t k4) {
    c10::cuda::CUDAGuard device_guard(a.device());
    auto stream = c10::cuda::getCurrentCUDAStream(a.get_device());
    // НАСТОЯЩИЙ ЧИСТЫЙ INT32 БЕЗ ПРОМЕЖУТОЧНОГО FLOAT32
    auto out = torch::empty({m, n}, a.options().dtype(torch::kInt32));
    dim3 block(16, 16);
    dim3 grid((n + 63) / 64, (m + 127) / 128);
    dp4a_mm_tiled_int32_kernel<<<grid, block, 0, stream>>>(
        a.data_ptr<int32_t>(), bt.data_ptr<int32_t>(), out.data_ptr<int32_t>(),
        static_cast<int>(m), static_cast<int>(n), static_cast<int>(k4)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor quantize_pack_affine(torch::Tensor x, torch::Tensor scale, torch::Tensor zero_point) {
    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
    auto out = torch::empty({x.numel() / 4}, x.options().dtype(torch::kInt32));
    constexpr int threads = 256;
    const int64_t groups = x.numel() / 4;
    const int blocks = static_cast<int>((groups + threads - 1) / threads);
    affine_quantize_pack_kernel<<<blocks, threads, 0, stream>>>(x.data_ptr<float>(), scale.data_ptr<float>(), zero_point.data_ptr<int32_t>(), out.data_ptr<int32_t>(), x.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}