#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>
#include <cmath>

// Аппаратная инструкция RDNA 2 v_dot4_i32_i8
__device__ __forceinline__ int32_t sdot4(int32_t a, int32_t b, int32_t acc) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return __builtin_amdgcn_sdot4(a, b, acc, false);
#else
    return acc;
#endif
}

// 1. Наивный кернел
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
    for (int p = 0; p < k4; ++p) {
        acc = sdot4(arow[p], brow[p], acc);
    }
    c[static_cast<int64_t>(row) * n + col] = acc;
}

__global__ void dp4a_mm_tiled_kernel(
    const int32_t* __restrict__ a,
    const int32_t* __restrict__ bt,
    int32_t* __restrict__ c,
    int m, int n, int k4)
{
    // Эксперимент: stride 17 вместо 20.
    // Рабочие значения K занимают столбцы 0..15.
    __shared__ int32_t a_tile[128][17];
    __shared__ int32_t b_tile[64][17];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    const int row0 = blockIdx.y * 128 + threadIdx.y * 8;
    const int col0 = blockIdx.x * 64 + threadIdx.x * 4;

    int32_t acc[8][4] = {0};

    // В каждой int4 находятся четыре упакованных INT8x4 слова.
    const int4* __restrict__ a_vec =
        reinterpret_cast<const int4*>(a);
    const int4* __restrict__ bt_vec =
        reinterpret_cast<const int4*>(bt);

    const int k_vec = k4 / 4;

    // A: два ряда на поток, четыре вектора по K на ряд.
    const int row_a0 = tid / 4;
    const int col_v_a = tid % 4;
    const int row_a1 = row_a0 + 64;

    const int ar0 = blockIdx.y * 128 + row_a0;
    const int ar1 = blockIdx.y * 128 + row_a1;

    // B: один ряд и один вектор на поток.
    const int row_b = tid / 4;
    const int col_v_b = tid % 4;
    const int br = blockIdx.x * 64 + row_b;

    for (int k0 = 0; k0 < k4; k0 += 16) {
        const int k_vec_base = k0 / 4 + col_v_a;

        int4 va0 =
            (ar0 < m && k_vec_base < k_vec)
            ? a_vec[static_cast<int64_t>(ar0) * k_vec + k_vec_base]
            : make_int4(0, 0, 0, 0);

        int4 va1 =
            (ar1 < m && k_vec_base < k_vec)
            ? a_vec[static_cast<int64_t>(ar1) * k_vec + k_vec_base]
            : make_int4(0, 0, 0, 0);

        const int a_col = col_v_a * 4;

        a_tile[row_a0][a_col + 0] = va0.x;
        a_tile[row_a0][a_col + 1] = va0.y;
        a_tile[row_a0][a_col + 2] = va0.z;
        a_tile[row_a0][a_col + 3] = va0.w;

        a_tile[row_a1][a_col + 0] = va1.x;
        a_tile[row_a1][a_col + 1] = va1.y;
        a_tile[row_a1][a_col + 2] = va1.z;
        a_tile[row_a1][a_col + 3] = va1.w;

        const int k_vec_base_b = k0 / 4 + col_v_b;

        int4 vb =
            (br < n && k_vec_base_b < k_vec)
            ? bt_vec[static_cast<int64_t>(br) * k_vec + k_vec_base_b]
            : make_int4(0, 0, 0, 0);

        const int b_col = col_v_b * 4;

        b_tile[row_b][b_col + 0] = vb.x;
        b_tile[row_b][b_col + 1] = vb.y;
        b_tile[row_b][b_col + 2] = vb.z;
        b_tile[row_b][b_col + 3] = vb.w;

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            int32_t a_vals[8];

            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                a_vals[r] =
                    a_tile[threadIdx.y * 8 + r][kk];
            }

            int32_t b_vals[4];

            #pragma unroll
            for (int ci = 0; ci < 4; ++ci) {
                b_vals[ci] =
                    b_tile[threadIdx.x * 4 + ci][kk];
            }

            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                #pragma unroll
                for (int ci = 0; ci < 4; ++ci) {
                    acc[r][ci] =
                        sdot4(a_vals[r], b_vals[ci], acc[r][ci]);
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        #pragma unroll
        for (int ci = 0; ci < 4; ++ci) {
            const int r_glob = row0 + r;
            const int c_glob = col0 + ci;

            if (r_glob < m && c_glob < n) {
                c[static_cast<int64_t>(r_glob) * n + c_glob] =
                    acc[r][ci];
            }
        }
    }
}

// 3. Квантователь
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

static void check_packed_inputs(const torch::Tensor& a, const torch::Tensor& bt, int64_t m, int64_t n, int64_t k4) {
    TORCH_CHECK(a.is_cuda() && bt.is_cuda(), "inputs must be GPU tensors (CUDA API on ROCm)");
    TORCH_CHECK(a.scalar_type() == torch::kInt32 && bt.scalar_type() == torch::kInt32, "packed inputs must be int32");
    TORCH_CHECK(a.is_contiguous() && bt.is_contiguous(), "packed inputs must be contiguous");
    TORCH_CHECK(a.dim() == 2 && bt.dim() == 2, "packed inputs must be 2D");
    TORCH_CHECK(a.size(0) == m && a.size(1) == k4, "A packed shape mismatch");
    TORCH_CHECK(bt.size(0) == n && bt.size(1) == k4, "B^T packed shape mismatch");
    TORCH_CHECK(a.device() == bt.device(), "inputs must be on same device");
    TORCH_CHECK(m > 0 && n > 0 && k4 > 0, "dimensions must be positive");
}

torch::Tensor dp4a_mm_naive(torch::Tensor a, torch::Tensor bt, int64_t m, int64_t n, int64_t k4) {
    check_packed_inputs(a, bt, m, n, k4);
    c10::cuda::CUDAGuard device_guard(a.device());
    auto out = torch::empty({m, n}, a.options().dtype(torch::kInt32));
    dim3 block(16, 16);
    dim3 grid((n + 15) / 16, (m + 15) / 16);
    dp4a_mm_naive_kernel<<<grid, block>>>(a.data_ptr<int32_t>(), bt.data_ptr<int32_t>(), out.data_ptr<int32_t>(), static_cast<int>(m), static_cast<int>(n), static_cast<int>(k4));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor dp4a_mm_tiled(torch::Tensor a, torch::Tensor bt, int64_t m, int64_t n, int64_t k4) {
    check_packed_inputs(a, bt, m, n, k4);
    TORCH_CHECK(m % 8 == 0 && n % 4 == 0 && k4 % 16 == 0, "tiled kernel requires M divisible by 8, N divisible by 4, and K divisible by 64 elements");
    c10::cuda::CUDAGuard device_guard(a.device());
    auto out = torch::empty({m, n}, a.options().dtype(torch::kInt32));
    dim3 block(16, 16);
    dim3 grid((n + 63) / 64, (m + 127) / 128);
    dp4a_mm_tiled_kernel<<<grid, block>>>(a.data_ptr<int32_t>(), bt.data_ptr<int32_t>(), out.data_ptr<int32_t>(), static_cast<int>(m), static_cast<int>(n), static_cast<int>(k4));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor quantize_pack_affine(torch::Tensor x, torch::Tensor scale, torch::Tensor zero_point) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32 && x.is_contiguous(), "x must be contiguous GPU float32");
    TORCH_CHECK(scale.is_cuda() && scale.scalar_type() == torch::kFloat32 && scale.numel() == 1, "scale must be a scalar GPU float32 tensor");
    TORCH_CHECK(zero_point.is_cuda() && zero_point.scalar_type() == torch::kInt32 && zero_point.numel() == 1, "zero_point must be a scalar GPU int32 tensor");
    TORCH_CHECK(x.numel() % 4 == 0, "x element count must be divisible by 4 for int8x4 packing");
    c10::cuda::CUDAGuard device_guard(x.device());
    auto out = torch::empty({x.numel() / 4}, x.options().dtype(torch::kInt32));
    constexpr int threads = 256;
    const int64_t groups = x.numel() / 4;
    const int blocks = static_cast<int>((groups + threads - 1) / threads);
    affine_quantize_pack_kernel<<<blocks, threads>>>(x.data_ptr<float>(), scale.data_ptr<float>(), zero_point.data_ptr<int32_t>(), out.data_ptr<int32_t>(), x.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}