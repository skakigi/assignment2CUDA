#include "sumcheck_native.h"

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>
#include <tuple>

namespace {

thread_local std::string g_last_error;

void set_error(const std::string& msg) {
    g_last_error = msg;
}

std::string cuda_error_message(const char* op, cudaError_t err) {
    std::ostringstream oss;
    oss << op << " failed: " << cudaGetErrorString(err);
    return oss.str();
}

inline bool cuda_ok(cudaError_t err, const char* op) {
    if (err == cudaSuccess) {
        return true;
    }
    set_error(cuda_error_message(op, err));
    return false;
}

template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;

    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr(other.ptr) {
        other.ptr = nullptr;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr != nullptr) {
                cudaFree(ptr);
            }
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }

    ~DeviceBuffer() {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }

    T* get() const {
        return ptr;
    }

    T** out() {
        return &ptr;
    }
};

enum class EvalKernelVariant {
    kBaselineShared = 0,
    kMleTiledShared = 1,
    kMleTiledWarp = 2,
    kMleTiledReduceIntrinsics = 3,
};

EvalKernelVariant select_eval_variant() {
    const char* env = std::getenv("SC_EVAL_VARIANT");
    if (env == nullptr) {
        return EvalKernelVariant::kBaselineShared;
    }
    const std::string value(env);
    if (value == "baseline" || value == "shared") {
        return EvalKernelVariant::kBaselineShared;
    }
    if (value == "mle_tiled" || value == "mle_tiled_shared" || value == "tile") {
        return EvalKernelVariant::kMleTiledShared;
    }
    if (value == "mle_tiled_warp" || value == "warp" || value == "shuffle") {
        return EvalKernelVariant::kMleTiledWarp;
    }
    if (value == "mle_tiled_reduce" || value == "reduce" || value == "intrinsics") {
        return EvalKernelVariant::kMleTiledReduceIntrinsics;
    }
    return EvalKernelVariant::kBaselineShared;
}

inline uint64_t make_barrett_mu_u32_host(uint32_t q) {
    if (q <= 1U) {
        return 0ULL;
    }
    return static_cast<uint64_t>((static_cast<unsigned __int128>(1) << 64) / q);
}

inline uint32_t add_mod_u32_host(uint32_t a, uint32_t b, uint32_t q) {
    if (q <= 1U) {
        return 0U;
    }
    uint64_t s = static_cast<uint64_t>(a) + static_cast<uint64_t>(b);
    if (s >= static_cast<uint64_t>(q)) {
        s -= static_cast<uint64_t>(q);
    }
    return static_cast<uint32_t>(s);
}

__device__ __forceinline__ uint32_t add_mod_u32_dev(uint32_t a, uint32_t b, uint32_t q) {
    if (q <= 1U) {
        return 0U;
    }
    uint64_t s = static_cast<uint64_t>(a) + static_cast<uint64_t>(b);
    if (s >= static_cast<uint64_t>(q)) {
        s -= static_cast<uint64_t>(q);
    }
    return static_cast<uint32_t>(s);
}

__device__ __forceinline__ uint32_t sub_mod_u32_dev(uint32_t a, uint32_t b, uint32_t q) {
    if (q <= 1U) {
        return 0U;
    }
    return (a >= b) ? (a - b) : static_cast<uint32_t>(static_cast<uint64_t>(a) + q - b);
}

__device__ __forceinline__ uint32_t mul_mod_u32_dev(
    uint32_t a,
    uint32_t b,
    uint32_t q,
    uint64_t q_recip) {
    if (q <= 1U) {
        return 0U;
    }
    const uint64_t prod = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
    const uint64_t qhat = __umul64hi(prod, q_recip);
    uint64_t rem = prod - qhat * static_cast<uint64_t>(q);
    if (rem >= static_cast<uint64_t>(q)) {
        rem -= static_cast<uint64_t>(q);
    }
    if (rem >= static_cast<uint64_t>(q)) {
        rem -= static_cast<uint64_t>(q);
    }
    return static_cast<uint32_t>(rem);
}

__device__ __forceinline__ uint32_t mle_update_u32_dev(
    uint32_t zero_eval,
    uint32_t one_eval,
    uint32_t target_eval,
    uint32_t q,
    uint64_t q_recip) {
    return add_mod_u32_dev(
        zero_eval,
        mul_mod_u32_dev(target_eval, sub_mod_u32_dev(one_eval, zero_eval, q), q, q_recip),
        q);
}

constexpr int kEvalTStride = 4;
constexpr int kMleItemsPerThread = 4;

inline int eval_threads_for_variant(EvalKernelVariant variant) {
    return (variant == EvalKernelVariant::kBaselineShared) ? 256 : 128;
}

inline uint64_t eval_items_per_block(EvalKernelVariant variant, int eval_threads) {
    if (variant == EvalKernelVariant::kBaselineShared) {
        return static_cast<uint64_t>(eval_threads);
    }
    return static_cast<uint64_t>(eval_threads) * static_cast<uint64_t>(kMleItemsPerThread);
}

inline size_t eval_shared_bytes_for_variant(
    EvalKernelVariant variant,
    int eval_threads,
    int32_t n_terms,
    int32_t total_term_vars) {
    const size_t metadata_bytes = static_cast<size_t>(n_terms + 1 + total_term_vars) * sizeof(int32_t);
    if (variant == EvalKernelVariant::kBaselineShared ||
        variant == EvalKernelVariant::kMleTiledShared) {
        return static_cast<size_t>(kEvalTStride) * static_cast<size_t>(eval_threads) * sizeof(uint32_t) +
               metadata_bytes;
    }
    const size_t warp_count = static_cast<size_t>((eval_threads + 31) / 32);
    if (variant == EvalKernelVariant::kMleTiledWarp) {
        return static_cast<size_t>(kEvalTStride) * warp_count * sizeof(uint32_t) + metadata_bytes;
    }
    return static_cast<size_t>(kEvalTStride) * warp_count * sizeof(uint64_t) + metadata_bytes;
}

__device__ __forceinline__ uint32_t warp_reduce_add_mod_u32(
    uint32_t value,
    uint32_t q,
    unsigned int mask) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = add_mod_u32_dev(value, __shfl_down_sync(mask, value, offset), q);
    }
    return value;
}

__device__ __forceinline__ uint64_t warp_reduce_add_u64_sync(
    uint64_t value,
    unsigned int mask) {
    // CUDA's __reduce_add_sync only supports 32-bit int/unsigned int.
    // For uint64_t reductions, use shuffle-based reduction explicitly.
    unsigned long long v = static_cast<unsigned long long>(value);
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset);
    }
    return static_cast<uint64_t>(v);
}

__device__ __forceinline__ uint32_t reduce_small_sum_mod_u32_dev(uint64_t value, uint32_t q) {
    if (q <= 1U) {
        return 0U;
    }
    while (value >= static_cast<uint64_t>(q)) {
        value -= static_cast<uint64_t>(q);
    }
    return static_cast<uint32_t>(value);
}

__global__ void eval_round_sums_u32_kernel(
    const uint32_t* const* __restrict__ tables,
    int32_t n_inputs,
    const int32_t* __restrict__ term_offsets,
    const int32_t* __restrict__ term_vars,
    int32_t n_terms,
    int32_t total_term_vars,
    uint64_t half,
    int32_t t_count,
    uint32_t q,
    uint64_t q_recip,
    uint32_t* __restrict__ block_sums_out) {
    extern __shared__ unsigned char shared_bytes[];

    uint32_t* const shared_sums = reinterpret_cast<uint32_t*>(shared_bytes);
    int32_t* const shared_term_offsets = reinterpret_cast<int32_t*>(shared_sums +
        static_cast<size_t>(kEvalTStride) * static_cast<size_t>(blockDim.x));
    int32_t* const shared_term_vars = shared_term_offsets + (static_cast<size_t>(n_terms) + 1U);

    for (int32_t idx = static_cast<int32_t>(threadIdx.x); idx < n_terms + 1; idx += blockDim.x) {
        shared_term_offsets[idx] = term_offsets[idx];
    }
    for (int32_t idx = static_cast<int32_t>(threadIdx.x); idx < total_term_vars; idx += blockDim.x) {
        shared_term_vars[idx] = term_vars[idx];
    }
    __syncthreads();

    const int32_t t_base = static_cast<int32_t>(blockIdx.y) * kEvalTStride;
    const uint64_t global_idx =
        static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x) +
        static_cast<uint64_t>(threadIdx.x);

    uint32_t point_values[kEvalTStride];
#pragma unroll
    for (int lane = 0; lane < kEvalTStride; ++lane) {
        point_values[lane] = 0U;
    }

    if (global_idx < half) {
        for (int32_t term = 0; term < n_terms; ++term) {
            uint32_t prod_values[kEvalTStride];
#pragma unroll
            for (int lane = 0; lane < kEvalTStride; ++lane) {
                prod_values[lane] = 1U;
            }

            const int32_t begin = shared_term_offsets[term];
            const int32_t end = shared_term_offsets[term + 1];
            for (int32_t j = begin; j < end; ++j) {
                const int32_t var_idx = shared_term_vars[j];
                if (var_idx < 0 || var_idx >= n_inputs) {
#pragma unroll
                    for (int lane = 0; lane < kEvalTStride; ++lane) {
                        prod_values[lane] = 0U;
                    }
                    break;
                }

                const uint32_t* const table = tables[var_idx];
                const uint32_t zero_eval = table[global_idx];
                const uint32_t one_eval = table[global_idx + half];
                const uint32_t delta_eval = sub_mod_u32_dev(one_eval, zero_eval, q);

#pragma unroll
                for (int lane = 0; lane < kEvalTStride; ++lane) {
                    const int32_t t_idx = t_base + lane;
                    if (t_idx < t_count) {
                        const uint32_t target_eval = static_cast<uint32_t>(t_idx);
                        const uint32_t folded_eval = add_mod_u32_dev(
                            zero_eval,
                            mul_mod_u32_dev(target_eval, delta_eval, q, q_recip),
                            q);
                        prod_values[lane] = mul_mod_u32_dev(prod_values[lane], folded_eval, q, q_recip);
                    }
                }
            }

#pragma unroll
            for (int lane = 0; lane < kEvalTStride; ++lane) {
                const int32_t t_idx = t_base + lane;
                if (t_idx < t_count) {
                    point_values[lane] = add_mod_u32_dev(point_values[lane], prod_values[lane], q);
                }
            }
        }
    }

#pragma unroll
    for (int lane = 0; lane < kEvalTStride; ++lane) {
        shared_sums[static_cast<size_t>(lane) * static_cast<size_t>(blockDim.x) + threadIdx.x] = point_values[lane];
    }
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
#pragma unroll
            for (int lane = 0; lane < kEvalTStride; ++lane) {
                const size_t base = static_cast<size_t>(lane) * static_cast<size_t>(blockDim.x);
                shared_sums[base + threadIdx.x] = add_mod_u32_dev(
                    shared_sums[base + threadIdx.x],
                    shared_sums[base + threadIdx.x + stride],
                    q);
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const uint32_t row_stride = static_cast<uint32_t>(gridDim.x);
#pragma unroll
        for (int lane = 0; lane < kEvalTStride; ++lane) {
            const int32_t t_idx = t_base + lane;
            if (t_idx < t_count) {
                const size_t base = static_cast<size_t>(lane) * static_cast<size_t>(blockDim.x);
                block_sums_out[static_cast<size_t>(t_idx) * row_stride + blockIdx.x] = shared_sums[base];
            }
        }
    }
}

template <bool UseWarpReduction>
__global__ void eval_round_sums_u32_mle_tiled_kernel(
    const uint32_t* const* __restrict__ tables,
    int32_t n_inputs,
    const int32_t* __restrict__ term_offsets,
    const int32_t* __restrict__ term_vars,
    int32_t n_terms,
    int32_t total_term_vars,
    uint64_t half,
    int32_t t_count,
    uint32_t q,
    uint64_t q_recip,
    uint32_t* __restrict__ block_sums_out) {
    extern __shared__ unsigned char shared_bytes[];

    const int warp_count = (static_cast<int>(blockDim.x) + 31) / 32;
    const size_t reduce_scratch_count = UseWarpReduction
        ? static_cast<size_t>(kEvalTStride) * static_cast<size_t>(warp_count)
        : static_cast<size_t>(kEvalTStride) * static_cast<size_t>(blockDim.x);

    uint32_t* const shared_reduce = reinterpret_cast<uint32_t*>(shared_bytes);
    int32_t* const shared_term_offsets = reinterpret_cast<int32_t*>(shared_reduce + reduce_scratch_count);
    int32_t* const shared_term_vars = shared_term_offsets + (static_cast<size_t>(n_terms) + 1U);

    for (int32_t idx = static_cast<int32_t>(threadIdx.x); idx < n_terms + 1; idx += blockDim.x) {
        shared_term_offsets[idx] = term_offsets[idx];
    }
    for (int32_t idx = static_cast<int32_t>(threadIdx.x); idx < total_term_vars; idx += blockDim.x) {
        shared_term_vars[idx] = term_vars[idx];
    }
    __syncthreads();

    const int32_t t_base = static_cast<int32_t>(blockIdx.y) * kEvalTStride;
    const uint64_t tile_base = static_cast<uint64_t>(blockIdx.x) *
        static_cast<uint64_t>(blockDim.x) * static_cast<uint64_t>(kMleItemsPerThread);

    uint32_t accum_values[kEvalTStride];
#pragma unroll
    for (int lane = 0; lane < kEvalTStride; ++lane) {
        accum_values[lane] = 0U;
    }

#pragma unroll
    for (int item = 0; item < kMleItemsPerThread; ++item) {
        const uint64_t global_idx = tile_base +
            static_cast<uint64_t>(item) * static_cast<uint64_t>(blockDim.x) +
            static_cast<uint64_t>(threadIdx.x);
        if (global_idx >= half) {
            continue;
        }

        for (int32_t term = 0; term < n_terms; ++term) {
            uint32_t prod_values[kEvalTStride];
#pragma unroll
            for (int lane = 0; lane < kEvalTStride; ++lane) {
                prod_values[lane] = 1U;
            }

            const int32_t begin = shared_term_offsets[term];
            const int32_t end = shared_term_offsets[term + 1];
            for (int32_t j = begin; j < end; ++j) {
                const int32_t var_idx = shared_term_vars[j];
                if (var_idx < 0 || var_idx >= n_inputs) {
#pragma unroll
                    for (int lane = 0; lane < kEvalTStride; ++lane) {
                        prod_values[lane] = 0U;
                    }
                    break;
                }

                const uint32_t* const table = tables[var_idx];
                const uint32_t zero_eval = table[global_idx];
                const uint32_t one_eval = table[global_idx + half];
                const uint32_t delta_eval = sub_mod_u32_dev(one_eval, zero_eval, q);

#pragma unroll
                for (int lane = 0; lane < kEvalTStride; ++lane) {
                    const int32_t t_idx = t_base + lane;
                    if (t_idx < t_count) {
                        const uint32_t target_eval = static_cast<uint32_t>(t_idx);
                        const uint32_t folded_eval = add_mod_u32_dev(
                            zero_eval,
                            mul_mod_u32_dev(target_eval, delta_eval, q, q_recip),
                            q);
                        prod_values[lane] = mul_mod_u32_dev(prod_values[lane], folded_eval, q, q_recip);
                    }
                }
            }

#pragma unroll
            for (int lane = 0; lane < kEvalTStride; ++lane) {
                const int32_t t_idx = t_base + lane;
                if (t_idx < t_count) {
                    accum_values[lane] = add_mod_u32_dev(accum_values[lane], prod_values[lane], q);
                }
            }
        }
    }

    if constexpr (!UseWarpReduction) {
#pragma unroll
        for (int lane = 0; lane < kEvalTStride; ++lane) {
            shared_reduce[static_cast<size_t>(lane) * static_cast<size_t>(blockDim.x) + threadIdx.x] = accum_values[lane];
        }
        __syncthreads();

        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
#pragma unroll
                for (int lane = 0; lane < kEvalTStride; ++lane) {
                    const size_t base = static_cast<size_t>(lane) * static_cast<size_t>(blockDim.x);
                    shared_reduce[base + threadIdx.x] = add_mod_u32_dev(
                        shared_reduce[base + threadIdx.x],
                        shared_reduce[base + threadIdx.x + stride],
                        q);
                }
            }
            __syncthreads();
        }
    } else {
        const int lane_id = static_cast<int>(threadIdx.x) & 31;
        const int warp_id = static_cast<int>(threadIdx.x) >> 5;
        const unsigned int full_mask = 0xffffffffu;

#pragma unroll
        for (int lane = 0; lane < kEvalTStride; ++lane) {
            accum_values[lane] = warp_reduce_add_mod_u32(accum_values[lane], q, full_mask);
        }

        if (lane_id == 0) {
#pragma unroll
            for (int lane = 0; lane < kEvalTStride; ++lane) {
                shared_reduce[static_cast<size_t>(lane) * static_cast<size_t>(warp_count) + warp_id] = accum_values[lane];
            }
        }
        __syncthreads();

        if (warp_id == 0) {
            uint32_t warp_values[kEvalTStride];
#pragma unroll
            for (int lane = 0; lane < kEvalTStride; ++lane) {
                warp_values[lane] = (lane_id < warp_count)
                    ? shared_reduce[static_cast<size_t>(lane) * static_cast<size_t>(warp_count) + lane_id]
                    : 0U;
            }
            const unsigned int active_mask = __ballot_sync(full_mask, lane_id < warp_count);
#pragma unroll
            for (int lane = 0; lane < kEvalTStride; ++lane) {
                warp_values[lane] = warp_reduce_add_mod_u32(warp_values[lane], q, active_mask);
            }
            if (lane_id == 0) {
#pragma unroll
                for (int lane = 0; lane < kEvalTStride; ++lane) {
                    shared_reduce[lane] = warp_values[lane];
                }
            }
        }
        __syncthreads();
#pragma unroll
        for (int lane = 0; lane < kEvalTStride; ++lane) {
            accum_values[lane] = shared_reduce[lane];
        }
    }

    if (threadIdx.x == 0) {
        const uint32_t row_stride = static_cast<uint32_t>(gridDim.x);
#pragma unroll
        for (int lane = 0; lane < kEvalTStride; ++lane) {
            const int32_t t_idx = t_base + lane;
            if (t_idx < t_count) {
                block_sums_out[static_cast<size_t>(t_idx) * row_stride + blockIdx.x] = accum_values[lane];
            }
        }
    }
}

__global__ void eval_round_sums_u32_mle_tiled_reduce_intrinsics_kernel(
    const uint32_t* const* __restrict__ tables,
    int32_t n_inputs,
    const int32_t* __restrict__ term_offsets,
    const int32_t* __restrict__ term_vars,
    int32_t n_terms,
    int32_t total_term_vars,
    uint64_t half,
    int32_t t_count,
    uint32_t q,
    uint64_t q_recip,
    uint32_t* __restrict__ block_sums_out) {
    extern __shared__ unsigned char shared_bytes[];

    const int warp_count = (static_cast<int>(blockDim.x) + 31) / 32;
    const size_t reduce_scratch_count = static_cast<size_t>(kEvalTStride) * static_cast<size_t>(warp_count);

    uint64_t* const shared_reduce = reinterpret_cast<uint64_t*>(shared_bytes);
    int32_t* const shared_term_offsets = reinterpret_cast<int32_t*>(shared_reduce + reduce_scratch_count);
    int32_t* const shared_term_vars = shared_term_offsets + (static_cast<size_t>(n_terms) + 1U);

    for (int32_t idx = static_cast<int32_t>(threadIdx.x); idx < n_terms + 1; idx += blockDim.x) {
        shared_term_offsets[idx] = term_offsets[idx];
    }
    for (int32_t idx = static_cast<int32_t>(threadIdx.x); idx < total_term_vars; idx += blockDim.x) {
        shared_term_vars[idx] = term_vars[idx];
    }
    __syncthreads();

    const int32_t t_base = static_cast<int32_t>(blockIdx.y) * kEvalTStride;
    const uint64_t tile_base = static_cast<uint64_t>(blockIdx.x) *
        static_cast<uint64_t>(blockDim.x) * static_cast<uint64_t>(kMleItemsPerThread);

    uint32_t accum_values[kEvalTStride];
#pragma unroll
    for (int lane = 0; lane < kEvalTStride; ++lane) {
        accum_values[lane] = 0U;
    }

#pragma unroll
    for (int item = 0; item < kMleItemsPerThread; ++item) {
        const uint64_t global_idx = tile_base +
            static_cast<uint64_t>(item) * static_cast<uint64_t>(blockDim.x) +
            static_cast<uint64_t>(threadIdx.x);
        if (global_idx >= half) {
            continue;
        }

        for (int32_t term = 0; term < n_terms; ++term) {
            uint32_t prod_values[kEvalTStride];
#pragma unroll
            for (int lane = 0; lane < kEvalTStride; ++lane) {
                prod_values[lane] = 1U;
            }

            const int32_t begin = shared_term_offsets[term];
            const int32_t end = shared_term_offsets[term + 1];
            for (int32_t j = begin; j < end; ++j) {
                const int32_t var_idx = shared_term_vars[j];
                if (var_idx < 0 || var_idx >= n_inputs) {
#pragma unroll
                    for (int lane = 0; lane < kEvalTStride; ++lane) {
                        prod_values[lane] = 0U;
                    }
                    break;
                }

                const uint32_t* const table = tables[var_idx];
                const uint32_t zero_eval = table[global_idx];
                const uint32_t one_eval = table[global_idx + half];
                const uint32_t delta_eval = sub_mod_u32_dev(one_eval, zero_eval, q);

#pragma unroll
                for (int lane = 0; lane < kEvalTStride; ++lane) {
                    const int32_t t_idx = t_base + lane;
                    if (t_idx < t_count) {
                        const uint32_t target_eval = static_cast<uint32_t>(t_idx);
                        const uint32_t folded_eval = add_mod_u32_dev(
                            zero_eval,
                            mul_mod_u32_dev(target_eval, delta_eval, q, q_recip),
                            q);
                        prod_values[lane] = mul_mod_u32_dev(prod_values[lane], folded_eval, q, q_recip);
                    }
                }
            }

#pragma unroll
            for (int lane = 0; lane < kEvalTStride; ++lane) {
                const int32_t t_idx = t_base + lane;
                if (t_idx < t_count) {
                    accum_values[lane] = add_mod_u32_dev(accum_values[lane], prod_values[lane], q);
                }
            }
        }
    }

    const int lane_id = static_cast<int>(threadIdx.x) & 31;
    const int warp_id = static_cast<int>(threadIdx.x) >> 5;
    const unsigned int full_mask = 0xffffffffu;

#pragma unroll
    for (int lane = 0; lane < kEvalTStride; ++lane) {
        const uint64_t warp_sum = warp_reduce_add_u64_sync(static_cast<uint64_t>(accum_values[lane]), full_mask);
        if (lane_id == 0) {
            shared_reduce[static_cast<size_t>(lane) * static_cast<size_t>(warp_count) + warp_id] = warp_sum;
        }
    }
    __syncthreads();

    uint32_t block_values[kEvalTStride];
#pragma unroll
    for (int lane = 0; lane < kEvalTStride; ++lane) {
        block_values[lane] = 0U;
    }

    if (warp_id == 0) {
        const unsigned int active_mask = __ballot_sync(full_mask, lane_id < warp_count);
#pragma unroll
        for (int lane = 0; lane < kEvalTStride; ++lane) {
            const uint64_t warp_value = (lane_id < warp_count)
                ? shared_reduce[static_cast<size_t>(lane) * static_cast<size_t>(warp_count) + lane_id]
                : 0ULL;
            const uint64_t block_sum = warp_reduce_add_u64_sync(warp_value, active_mask);
            if (lane_id == 0) {
                block_values[lane] = reduce_small_sum_mod_u32_dev(block_sum, q);
            }
        }
    }

    if (threadIdx.x == 0) {
        const uint32_t row_stride = static_cast<uint32_t>(gridDim.x);
#pragma unroll
        for (int lane = 0; lane < kEvalTStride; ++lane) {
            const int32_t t_idx = t_base + lane;
            if (t_idx < t_count) {
                block_sums_out[static_cast<size_t>(t_idx) * row_stride + blockIdx.x] = block_values[lane];
            }
        }
    }
}

__global__ void reduce_rows_u32_intrinsics_kernel(
    const uint32_t* input_sums,
    uint32_t* output_sums,
    uint64_t row_width,
    int32_t n_rows,
    uint32_t q) {
    extern __shared__ unsigned char shared_bytes[];
    uint64_t* const shared_reduce = reinterpret_cast<uint64_t*>(shared_bytes);

    const int32_t row_idx = static_cast<int32_t>(blockIdx.y);
    if (row_idx >= n_rows) {
        return;
    }

    const uint64_t global_idx =
        static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x) +
        static_cast<uint64_t>(threadIdx.x);

    uint32_t value = 0U;
    if (global_idx < row_width) {
        value = input_sums[static_cast<size_t>(row_idx) * row_width + global_idx];
    }

    const int lane_id = static_cast<int>(threadIdx.x) & 31;
    const int warp_id = static_cast<int>(threadIdx.x) >> 5;
    const int warp_count = (static_cast<int>(blockDim.x) + 31) / 32;
    const unsigned int full_mask = 0xffffffffu;

    const uint64_t warp_sum = warp_reduce_add_u64_sync(static_cast<uint64_t>(value), full_mask);
    if (lane_id == 0) {
        shared_reduce[warp_id] = warp_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        const unsigned int active_mask = __ballot_sync(full_mask, lane_id < warp_count);
        const uint64_t block_input = (lane_id < warp_count) ? shared_reduce[lane_id] : 0ULL;
        const uint64_t block_sum = warp_reduce_add_u64_sync(block_input, active_mask);
        if (lane_id == 0) {
            const uint32_t out_row_stride = static_cast<uint32_t>(gridDim.x);
            output_sums[static_cast<size_t>(row_idx) * out_row_stride + blockIdx.x] =
                reduce_small_sum_mod_u32_dev(block_sum, q);
        }
    }
}

__global__ void reduce_rows_u32_kernel(
    const uint32_t* input_sums,
    uint32_t* output_sums,
    uint64_t row_width,
    int32_t n_rows,
    uint32_t q) {
    extern __shared__ uint32_t shared_sums[];

    const int32_t row_idx = static_cast<int32_t>(blockIdx.y);
    if (row_idx >= n_rows) {
        return;
    }

    const uint64_t global_idx =
        static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x) +
        static_cast<uint64_t>(threadIdx.x);

    uint32_t value = 0;
    if (global_idx < row_width) {
        value = input_sums[static_cast<size_t>(row_idx) * row_width + global_idx];
    }

    shared_sums[threadIdx.x] = value;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sums[threadIdx.x] = add_mod_u32_dev(
                shared_sums[threadIdx.x],
                shared_sums[threadIdx.x + stride],
                q);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const uint32_t out_row_stride = static_cast<uint32_t>(gridDim.x);
        output_sums[static_cast<size_t>(row_idx) * out_row_stride + blockIdx.x] = shared_sums[0];
    }
}

__global__ void fold_tables_in_place_u32_kernel(
    uint32_t* const* __restrict__ tables,
    int32_t n_inputs,
    uint64_t half,
    uint32_t target_eval,
    uint32_t q,
    uint64_t q_recip) {
    const uint64_t local_idx =
        static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x) +
        static_cast<uint64_t>(threadIdx.x);
    const int32_t input_idx = static_cast<int32_t>(blockIdx.y);
    if (input_idx >= n_inputs || local_idx >= half) {
        return;
    }

    uint32_t* const table = tables[input_idx];
    const uint32_t zero_eval = table[local_idx];
    const uint32_t one_eval = table[local_idx + half];
    table[local_idx] = mle_update_u32_dev(zero_eval, one_eval, target_eval, q, q_recip);
}


__global__ void set_claim0_from_round0_u32_kernel(
    const uint32_t* __restrict__ round0,
    uint32_t* __restrict__ claim0,
    uint32_t q) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        claim0[0] = add_mod_u32_dev(round0[0], round0[1], q);
    }
}

int validate_inputs(
    const uint32_t* const* host_inputs,
    int32_t n_inputs,
    const int32_t* term_offsets,
    const int32_t* term_vars,
    int32_t n_terms,
    uint64_t n,
    uint32_t q,
    const uint32_t* host_challenges,
    int32_t n_prover_challenges,
    int32_t num_rounds,
    uint32_t* host_claim0_out,
    uint32_t* host_round_evals_out) {
    if (n_inputs <= 0 || n_terms <= 0 || n == 0 || num_rounds <= 0) {
        set_error("invalid sizes passed to sc_sumcheck_u32");
        return 1;
    }
    if (q == 0) {
        set_error("q must be non-zero");
        return 1;
    }
    if (host_inputs == nullptr || term_offsets == nullptr || term_vars == nullptr ||
        host_claim0_out == nullptr || host_round_evals_out == nullptr) {
        set_error("null pointer passed to sc_sumcheck_u32");
        return 1;
    }
    if (n_prover_challenges > 0 && host_challenges == nullptr) {
        set_error("host_challenges is null but prover challenges were requested");
        return 1;
    }
    if (n != (1ULL << num_rounds)) {
        set_error("n must equal 2**num_rounds in this scaffold");
        return 1;
    }
    if (n_prover_challenges != std::max(0, num_rounds - 1)) {
        set_error("expected num_rounds-1 prover challenges");
        return 1;
    }
    for (int32_t i = 0; i < n_inputs; ++i) {
        if (host_inputs[i] == nullptr) {
            set_error("one of the input pointers is null");
            return 1;
        }
    }
    if (term_offsets[0] != 0) {
        set_error("term_offsets must start at 0");
        return 1;
    }
    for (int32_t term = 0; term < n_terms; ++term) {
        if (term_offsets[term] > term_offsets[term + 1]) {
            set_error("term_offsets must be non-decreasing");
            return 1;
        }
    }
    for (int32_t j = 0; j < term_offsets[n_terms]; ++j) {
        if (term_vars[j] < 0 || term_vars[j] >= n_inputs) {
            set_error("term_vars contains an out-of-range input index");
            return 1;
        }
    }
    return 0;
}

int gpu_sumcheck_u32(
    const uint32_t* const* host_inputs,
    int32_t n_inputs,
    const int32_t* term_offsets,
    const int32_t* term_vars,
    int32_t n_terms,
    uint64_t n,
    uint32_t q,
    const uint32_t* host_challenges,
    int32_t n_prover_challenges,
    int32_t num_rounds,
    uint32_t* host_claim0_out,
    uint32_t* host_round_evals_out,
    ScRunTimings* timings_out) {
    int rc = validate_inputs(
        host_inputs,
        n_inputs,
        term_offsets,
        term_vars,
        n_terms,
        n,
        q,
        host_challenges,
        n_prover_challenges,
        num_rounds,
        host_claim0_out,
        host_round_evals_out);
    if (rc != 0) {
        return rc;
    }

    const auto total_start = std::chrono::high_resolution_clock::now();
    double h2d_ms = 0.0;
    double kernel_ms = 0.0;
    double d2h_ms = 0.0;

    int32_t max_degree = 0;
    for (int32_t term = 0; term < n_terms; ++term) {
        max_degree = std::max(max_degree, term_offsets[term + 1] - term_offsets[term]);
    }
    if (max_degree < 1) {
        set_error("sumcheck requires max degree at least 1 to form g(0) and g(1)");
        return 1;
    }

    const int32_t total_term_vars = term_offsets[n_terms];
    const int32_t t_count = max_degree + 1;
    const uint64_t q_recip = make_barrett_mu_u32_host(q);
    const EvalKernelVariant eval_variant = select_eval_variant();
    const int eval_threads = eval_threads_for_variant(eval_variant);
    const uint64_t eval_block_items = eval_items_per_block(eval_variant, eval_threads);
    const int reduce_threads = 256;

    std::vector<DeviceBuffer<uint32_t>> d_tables(static_cast<size_t>(n_inputs));
    DeviceBuffer<uint32_t*> d_table_ptrs;
    DeviceBuffer<int32_t> d_term_offsets;
    DeviceBuffer<int32_t> d_term_vars;
    DeviceBuffer<uint32_t> d_block_sums_a;
    DeviceBuffer<uint32_t> d_block_sums_b;
    std::vector<uint32_t*> h_table_ptrs(static_cast<size_t>(n_inputs), nullptr);
    size_t block_sums_capacity = 0;

    {
        const auto phase_start = std::chrono::high_resolution_clock::now();
        for (int32_t i = 0; i < n_inputs; ++i) {
            if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(d_tables[i].out()), n * sizeof(uint32_t)), "cudaMalloc(input_table)")) {
                return 1;
            }
            if (!cuda_ok(cudaMemcpy(d_tables[i].get(), host_inputs[i], n * sizeof(uint32_t), cudaMemcpyHostToDevice), "cudaMemcpy(input_table H2D)")) {
                return 1;
            }
            h_table_ptrs[i] = d_tables[i].get();
        }
        if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(d_table_ptrs.out()), static_cast<size_t>(n_inputs) * sizeof(uint32_t*)), "cudaMalloc(table_ptrs)")) {
            return 1;
        }
        if (!cuda_ok(cudaMemcpy(d_table_ptrs.get(), h_table_ptrs.data(), static_cast<size_t>(n_inputs) * sizeof(uint32_t*), cudaMemcpyHostToDevice), "cudaMemcpy(table_ptrs H2D)")) {
            return 1;
        }
        if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(d_term_offsets.out()), static_cast<size_t>(n_terms + 1) * sizeof(int32_t)), "cudaMalloc(term_offsets)")) {
            return 1;
        }
        if (!cuda_ok(cudaMemcpy(d_term_offsets.get(), term_offsets, static_cast<size_t>(n_terms + 1) * sizeof(int32_t), cudaMemcpyHostToDevice), "cudaMemcpy(term_offsets H2D)")) {
            return 1;
        }
        if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(d_term_vars.out()), static_cast<size_t>(total_term_vars) * sizeof(int32_t)), "cudaMalloc(term_vars)")) {
            return 1;
        }
        if (!cuda_ok(cudaMemcpy(d_term_vars.get(), term_vars, static_cast<size_t>(total_term_vars) * sizeof(int32_t), cudaMemcpyHostToDevice), "cudaMemcpy(term_vars H2D)")) {
            return 1;
        }
        h2d_ms += std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - phase_start).count();
    }

    uint64_t current_len = n;
    for (int32_t round = 0; round < num_rounds; ++round) {
        const uint64_t half = current_len / 2;
        if (half == 0) {
            set_error("encountered zero half-length during sumcheck");
            return 1;
        }

        const int blocks_eval = static_cast<int>((half + eval_block_items - 1ULL) / eval_block_items);
        const size_t needed_block_sums = static_cast<size_t>(t_count) *
                                         static_cast<size_t>(blocks_eval);
        if (needed_block_sums > block_sums_capacity) {
            d_block_sums_a = DeviceBuffer<uint32_t>();
            d_block_sums_b = DeviceBuffer<uint32_t>();
            if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(d_block_sums_a.out()),
                                    needed_block_sums * sizeof(uint32_t)),
                         "cudaMalloc(block_sums_a)")) {
                return 1;
            }
            if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(d_block_sums_b.out()),
                                    needed_block_sums * sizeof(uint32_t)),
                         "cudaMalloc(block_sums_b)")) {
                return 1;
            }
            block_sums_capacity = needed_block_sums;
        }

        uint32_t* final_round_sums_ptr = d_block_sums_a.get();
        {
            const auto phase_start = std::chrono::high_resolution_clock::now();
            const int t_tiles = (t_count + kEvalTStride - 1) / kEvalTStride;
            const dim3 eval_grid(static_cast<unsigned int>(blocks_eval),
                                 static_cast<unsigned int>(t_tiles),
                                 1U);
            const size_t eval_shared_bytes = eval_shared_bytes_for_variant(
                eval_variant,
                eval_threads,
                n_terms,
                total_term_vars);

            switch (eval_variant) {
                case EvalKernelVariant::kBaselineShared:
                    eval_round_sums_u32_kernel<<<eval_grid, eval_threads, eval_shared_bytes>>>(
                        reinterpret_cast<const uint32_t* const*>(d_table_ptrs.get()),
                        n_inputs,
                        d_term_offsets.get(),
                        d_term_vars.get(),
                        n_terms,
                        total_term_vars,
                        half,
                        t_count,
                        q,
                        q_recip,
                        d_block_sums_a.get());
                    if (!cuda_ok(cudaGetLastError(), "eval_round_sums_u32_kernel launch")) {
                        return 1;
                    }
                    break;
                case EvalKernelVariant::kMleTiledShared:
                    eval_round_sums_u32_mle_tiled_kernel<false><<<eval_grid, eval_threads, eval_shared_bytes>>>(
                        reinterpret_cast<const uint32_t* const*>(d_table_ptrs.get()),
                        n_inputs,
                        d_term_offsets.get(),
                        d_term_vars.get(),
                        n_terms,
                        total_term_vars,
                        half,
                        t_count,
                        q,
                        q_recip,
                        d_block_sums_a.get());
                    if (!cuda_ok(cudaGetLastError(), "eval_round_sums_u32_mle_tiled_kernel(shared) launch")) {
                        return 1;
                    }
                    break;
                case EvalKernelVariant::kMleTiledWarp:
                    eval_round_sums_u32_mle_tiled_kernel<true><<<eval_grid, eval_threads, eval_shared_bytes>>>(
                        reinterpret_cast<const uint32_t* const*>(d_table_ptrs.get()),
                        n_inputs,
                        d_term_offsets.get(),
                        d_term_vars.get(),
                        n_terms,
                        total_term_vars,
                        half,
                        t_count,
                        q,
                        q_recip,
                        d_block_sums_a.get());
                    if (!cuda_ok(cudaGetLastError(), "eval_round_sums_u32_mle_tiled_kernel(warp) launch")) {
                        return 1;
                    }
                    break;
                case EvalKernelVariant::kMleTiledReduceIntrinsics:
                    eval_round_sums_u32_mle_tiled_reduce_intrinsics_kernel<<<eval_grid, eval_threads, eval_shared_bytes>>>(
                        reinterpret_cast<const uint32_t* const*>(d_table_ptrs.get()),
                        n_inputs,
                        d_term_offsets.get(),
                        d_term_vars.get(),
                        n_terms,
                        total_term_vars,
                        half,
                        t_count,
                        q,
                        q_recip,
                        d_block_sums_a.get());
                    if (!cuda_ok(cudaGetLastError(), "eval_round_sums_u32_mle_tiled_reduce_intrinsics_kernel launch")) {
                        return 1;
                    }
                    break;
            }

            uint64_t reduce_width = static_cast<uint64_t>(blocks_eval);
            uint32_t* reduce_in = d_block_sums_a.get();
            uint32_t* reduce_out = d_block_sums_b.get();
            while (reduce_width > 1) {
                const int reduce_blocks = static_cast<int>((reduce_width + static_cast<uint64_t>(reduce_threads) - 1ULL) /
                                                           static_cast<uint64_t>(reduce_threads));
                const dim3 reduce_grid(static_cast<unsigned int>(reduce_blocks),
                                       static_cast<unsigned int>(t_count),
                                       1U);
                if (eval_variant == EvalKernelVariant::kMleTiledReduceIntrinsics) {
                    const size_t reduce_shared_bytes =
                        static_cast<size_t>((reduce_threads + 31) / 32) * sizeof(uint64_t);
                    reduce_rows_u32_intrinsics_kernel<<<reduce_grid, reduce_threads, reduce_shared_bytes>>>(
                        reduce_in,
                        reduce_out,
                        reduce_width,
                        t_count,
                        q);
                    if (!cuda_ok(cudaGetLastError(), "reduce_rows_u32_intrinsics_kernel launch")) {
                        return 1;
                    }
                } else {
                    reduce_rows_u32_kernel<<<reduce_grid, reduce_threads,
                                             static_cast<size_t>(reduce_threads) * sizeof(uint32_t)>>>(
                        reduce_in,
                        reduce_out,
                        reduce_width,
                        t_count,
                        q);
                    if (!cuda_ok(cudaGetLastError(), "reduce_rows_u32_kernel launch")) {
                        return 1;
                    }
                }
                reduce_width = static_cast<uint64_t>(reduce_blocks);
                std::swap(reduce_in, reduce_out);
            }
            final_round_sums_ptr = reduce_in;

            if (!cuda_ok(cudaDeviceSynchronize(), "round row reduction sync")) {
                return 1;
            }
            kernel_ms += std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - phase_start).count();
        }

        {
            const auto phase_start = std::chrono::high_resolution_clock::now();
            const size_t row_offset = static_cast<size_t>(round) * static_cast<size_t>(t_count);
            if (!cuda_ok(cudaMemcpy(host_round_evals_out + row_offset,
                                    final_round_sums_ptr,
                                    static_cast<size_t>(t_count) * sizeof(uint32_t),
                                    cudaMemcpyDeviceToHost),
                         "cudaMemcpy(round_row D2H)")) {
                return 1;
            }
            d2h_ms += std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - phase_start).count();
        }

        if (round == 0) {
            *host_claim0_out = add_mod_u32_host(host_round_evals_out[0], host_round_evals_out[1], q);
        }

        if (round < n_prover_challenges) {
            const uint32_t r = host_challenges[round] % q;
            const int blocks_fold = static_cast<int>((half + static_cast<uint64_t>(reduce_threads) - 1ULL) /
                                                     static_cast<uint64_t>(reduce_threads));
            const auto phase_start = std::chrono::high_resolution_clock::now();
            const dim3 fold_grid(static_cast<unsigned int>(blocks_fold),
                                 static_cast<unsigned int>(n_inputs),
                                 1U);
            fold_tables_in_place_u32_kernel<<<fold_grid, reduce_threads>>>(
                d_table_ptrs.get(),
                n_inputs,
                half,
                r,
                q,
                q_recip);
            if (!cuda_ok(cudaGetLastError(), "fold_tables_in_place_u32_kernel launch")) {
                return 1;
            }
            if (!cuda_ok(cudaDeviceSynchronize(), "fold_tables_in_place_u32_kernel sync")) {
                return 1;
            }
            kernel_ms += std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - phase_start).count();
        }

        current_len = half;
    }

    const auto total_end = std::chrono::high_resolution_clock::now();
    if (timings_out != nullptr) {
        timings_out->h2d_ms = static_cast<float>(h2d_ms);
        timings_out->kernel_ms = static_cast<float>(kernel_ms);
        timings_out->d2h_ms = static_cast<float>(d2h_ms);
        timings_out->total_ms = static_cast<float>(
            std::chrono::duration<double, std::milli>(total_end - total_start).count());
    }
    return 0;
}


std::tuple<torch::Tensor, torch::Tensor> torch_sumcheck_terms_u32_cuda(
    torch::Tensor eval_tables,
    torch::Tensor challenges,
    torch::Tensor term_offsets,
    torch::Tensor term_vars,
    uint32_t q) {
    if (!eval_tables.is_cuda()) {
        throw std::invalid_argument("eval_tables must be a CUDA tensor");
    }
    if (eval_tables.dim() != 2) {
        throw std::invalid_argument("eval_tables must have shape (n_inputs, n)");
    }
    if (eval_tables.scalar_type() != torch::kUInt32 && eval_tables.scalar_type() != torch::kInt32) {
        throw std::invalid_argument("eval_tables must have dtype torch.uint32 or torch.int32");
    }
    if (challenges.dim() != 1) {
        throw std::invalid_argument("challenges must be 1D");
    }
    if (term_offsets.dim() != 1 || term_vars.dim() != 1) {
        throw std::invalid_argument("term_offsets and term_vars must be 1D");
    }
    if (q == 0U) {
        throw std::invalid_argument("q must be non-zero");
    }

    const c10::cuda::CUDAGuard device_guard(eval_tables.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int32_t n_inputs = static_cast<int32_t>(eval_tables.size(0));
    const uint64_t n = static_cast<uint64_t>(eval_tables.size(1));
    if (n == 0 || (n & (n - 1ULL)) != 0ULL) {
        throw std::invalid_argument("n must be a nonzero power of two");
    }
    int32_t num_rounds = 0;
    for (uint64_t tmp = n; tmp > 1ULL; tmp >>= 1ULL) {
        ++num_rounds;
    }

    auto term_offsets_cpu = term_offsets.to(torch::kCPU, torch::kInt32).contiguous();
    const int32_t n_terms = static_cast<int32_t>(term_offsets_cpu.numel()) - 1;
    if (n_terms <= 0) {
        throw std::invalid_argument("term_offsets must have length >= 2");
    }
    const int32_t* term_offsets_cpu_ptr = term_offsets_cpu.data_ptr<int32_t>();
    if (term_offsets_cpu_ptr[0] != 0) {
        throw std::invalid_argument("term_offsets must start at 0");
    }
    const int32_t total_term_vars = term_offsets_cpu_ptr[n_terms];
    if (total_term_vars < 0 || total_term_vars != term_vars.numel()) {
        throw std::invalid_argument("term_offsets[-1] must equal len(term_vars)");
    }
    int32_t max_degree = 0;
    for (int32_t term = 0; term < n_terms; ++term) {
        if (term_offsets_cpu_ptr[term] > term_offsets_cpu_ptr[term + 1]) {
            throw std::invalid_argument("term_offsets must be non-decreasing");
        }
        max_degree = std::max(max_degree, term_offsets_cpu_ptr[term + 1] - term_offsets_cpu_ptr[term]);
    }
    if (max_degree < 1) {
        throw std::invalid_argument("max expression degree must be at least 1");
    }
    const int32_t t_count = max_degree + 1;

    auto term_vars_cpu = term_vars.to(torch::kCPU, torch::kInt32).contiguous();
    const int32_t* term_vars_cpu_ptr = term_vars_cpu.data_ptr<int32_t>();
    for (int32_t i = 0; i < total_term_vars; ++i) {
        if (term_vars_cpu_ptr[i] < 0 || term_vars_cpu_ptr[i] >= n_inputs) {
            throw std::invalid_argument("term_vars contains out-of-range input index");
        }
    }

    auto rs_cpu = challenges.to(torch::kCPU, torch::kUInt32).contiguous();
    const int64_t available_challenges = rs_cpu.numel();
    if (available_challenges < std::max(0, static_cast<int>(num_rounds - 1))) {
        throw std::invalid_argument("need at least num_rounds-1 prover challenges");
    }

    auto options = eval_tables.options();
    auto current = eval_tables.contiguous().clone();
    auto round_evals = torch::empty({num_rounds, t_count}, options);
    auto claim0 = torch::empty({1}, options);

    std::vector<uint32_t*> h_table_ptrs(static_cast<size_t>(n_inputs), nullptr);
    uint32_t* base_ptr = reinterpret_cast<uint32_t*>(current.data_ptr());
    for (int32_t i = 0; i < n_inputs; ++i) {
        h_table_ptrs[static_cast<size_t>(i)] = base_ptr + static_cast<size_t>(i) * static_cast<size_t>(n);
    }

    DeviceBuffer<uint32_t*> d_table_ptrs;
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(d_table_ptrs.out()), static_cast<size_t>(n_inputs) * sizeof(uint32_t*)), "cudaMalloc(torch table_ptrs)")) {
        throw std::runtime_error(g_last_error);
    }
    if (!cuda_ok(cudaMemcpyAsync(d_table_ptrs.get(), h_table_ptrs.data(), static_cast<size_t>(n_inputs) * sizeof(uint32_t*), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(torch table_ptrs H2D)")) {
        throw std::runtime_error(g_last_error);
    }

    auto d_term_offsets_t = term_offsets_cpu.to(eval_tables.device(), torch::kInt32, false, true);
    auto d_term_vars_t = term_vars_cpu.to(eval_tables.device(), torch::kInt32, false, true);

    const uint64_t q_recip = make_barrett_mu_u32_host(q);
    const EvalKernelVariant eval_variant = select_eval_variant();
    const int eval_threads = eval_threads_for_variant(eval_variant);
    const uint64_t eval_block_items = eval_items_per_block(eval_variant, eval_threads);
    const int reduce_threads = 256;

    uint64_t current_len = n;
    for (int32_t round = 0; round < num_rounds; ++round) {
        const uint64_t half = current_len / 2ULL;
        const int blocks_eval = static_cast<int>((half + eval_block_items - 1ULL) / eval_block_items);
        const int t_tiles = (t_count + kEvalTStride - 1) / kEvalTStride;
        const dim3 eval_grid(static_cast<unsigned int>(blocks_eval), static_cast<unsigned int>(t_tiles), 1U);
        const size_t eval_shared_bytes = eval_shared_bytes_for_variant(eval_variant, eval_threads, n_terms, total_term_vars);

        auto d_block_sums_a = torch::empty({t_count, blocks_eval}, options);
        auto d_block_sums_b = torch::empty({t_count, blocks_eval}, options);
        uint32_t* a_ptr = reinterpret_cast<uint32_t*>(d_block_sums_a.data_ptr());
        uint32_t* b_ptr = reinterpret_cast<uint32_t*>(d_block_sums_b.data_ptr());

        switch (eval_variant) {
            case EvalKernelVariant::kBaselineShared:
                eval_round_sums_u32_kernel<<<eval_grid, eval_threads, eval_shared_bytes, stream>>>(
                    reinterpret_cast<const uint32_t* const*>(d_table_ptrs.get()),
                    n_inputs,
                    d_term_offsets_t.data_ptr<int32_t>(),
                    d_term_vars_t.data_ptr<int32_t>(),
                    n_terms,
                    total_term_vars,
                    half,
                    t_count,
                    q,
                    q_recip,
                    a_ptr);
                break;
            case EvalKernelVariant::kMleTiledShared:
                eval_round_sums_u32_mle_tiled_kernel<false><<<eval_grid, eval_threads, eval_shared_bytes, stream>>>(
                    reinterpret_cast<const uint32_t* const*>(d_table_ptrs.get()),
                    n_inputs,
                    d_term_offsets_t.data_ptr<int32_t>(),
                    d_term_vars_t.data_ptr<int32_t>(),
                    n_terms,
                    total_term_vars,
                    half,
                    t_count,
                    q,
                    q_recip,
                    a_ptr);
                break;
            case EvalKernelVariant::kMleTiledWarp:
                eval_round_sums_u32_mle_tiled_kernel<true><<<eval_grid, eval_threads, eval_shared_bytes, stream>>>(
                    reinterpret_cast<const uint32_t* const*>(d_table_ptrs.get()),
                    n_inputs,
                    d_term_offsets_t.data_ptr<int32_t>(),
                    d_term_vars_t.data_ptr<int32_t>(),
                    n_terms,
                    total_term_vars,
                    half,
                    t_count,
                    q,
                    q_recip,
                    a_ptr);
                break;
            case EvalKernelVariant::kMleTiledReduceIntrinsics:
                eval_round_sums_u32_mle_tiled_reduce_intrinsics_kernel<<<eval_grid, eval_threads, eval_shared_bytes, stream>>>(
                    reinterpret_cast<const uint32_t* const*>(d_table_ptrs.get()),
                    n_inputs,
                    d_term_offsets_t.data_ptr<int32_t>(),
                    d_term_vars_t.data_ptr<int32_t>(),
                    n_terms,
                    total_term_vars,
                    half,
                    t_count,
                    q,
                    q_recip,
                    a_ptr);
                break;
        }
        if (!cuda_ok(cudaGetLastError(), "torch eval_round launch")) {
            throw std::runtime_error(g_last_error);
        }

        uint64_t reduce_width = static_cast<uint64_t>(blocks_eval);
        uint32_t* reduce_in = a_ptr;
        uint32_t* reduce_out = b_ptr;
        while (reduce_width > 1ULL) {
            const int reduce_blocks = static_cast<int>((reduce_width + static_cast<uint64_t>(reduce_threads) - 1ULL) / static_cast<uint64_t>(reduce_threads));
            const dim3 reduce_grid(static_cast<unsigned int>(reduce_blocks), static_cast<unsigned int>(t_count), 1U);
            if (eval_variant == EvalKernelVariant::kMleTiledReduceIntrinsics) {
                const size_t reduce_shared_bytes = static_cast<size_t>((reduce_threads + 31) / 32) * sizeof(uint64_t);
                reduce_rows_u32_intrinsics_kernel<<<reduce_grid, reduce_threads, reduce_shared_bytes, stream>>>(
                    reduce_in, reduce_out, reduce_width, t_count, q);
            } else {
                reduce_rows_u32_kernel<<<reduce_grid, reduce_threads, static_cast<size_t>(reduce_threads) * sizeof(uint32_t), stream>>>(
                    reduce_in, reduce_out, reduce_width, t_count, q);
            }
            if (!cuda_ok(cudaGetLastError(), "torch reduce_rows launch")) {
                throw std::runtime_error(g_last_error);
            }
            reduce_width = static_cast<uint64_t>(reduce_blocks);
            std::swap(reduce_in, reduce_out);
        }

        uint32_t* out_row = reinterpret_cast<uint32_t*>(round_evals[round].data_ptr());
        if (!cuda_ok(cudaMemcpyAsync(out_row, reduce_in, static_cast<size_t>(t_count) * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync round evals D2D")) {
            throw std::runtime_error(g_last_error);
        }
        if (round == 0) {
            set_claim0_from_round0_u32_kernel<<<1, 1, 0, stream>>>(
                reduce_in,
                reinterpret_cast<uint32_t*>(claim0.data_ptr()),
                q);
            if (!cuda_ok(cudaGetLastError(), "set_claim0_from_round0_u32_kernel launch")) {
                throw std::runtime_error(g_last_error);
            }
        }

        if (round < num_rounds - 1) {
            const uint32_t r = rs_cpu.data_ptr<uint32_t>()[round] % q;
            const int blocks_fold = static_cast<int>((half + static_cast<uint64_t>(reduce_threads) - 1ULL) / static_cast<uint64_t>(reduce_threads));
            const dim3 fold_grid(static_cast<unsigned int>(blocks_fold), static_cast<unsigned int>(n_inputs), 1U);
            fold_tables_in_place_u32_kernel<<<fold_grid, reduce_threads, 0, stream>>>(
                d_table_ptrs.get(),
                n_inputs,
                half,
                r,
                q,
                q_recip);
            if (!cuda_ok(cudaGetLastError(), "torch fold_tables launch")) {
                throw std::runtime_error(g_last_error);
            }
        }
        current_len = half;
    }

    return std::make_tuple(claim0, round_evals);
}

}  // namespace

extern "C" int sc_sumcheck_u32(
    const uint32_t* const* host_inputs,
    int32_t n_inputs,
    const int32_t* term_offsets,
    const int32_t* term_vars,
    int32_t n_terms,
    uint64_t n,
    uint32_t q,
    const uint32_t* host_challenges,
    int32_t n_prover_challenges,
    int32_t num_rounds,
    uint32_t* host_claim0_out,
    uint32_t* host_round_evals_out,
    ScRunTimings* timings_out) {
    g_last_error.clear();
    return gpu_sumcheck_u32(
        host_inputs,
        n_inputs,
        term_offsets,
        term_vars,
        n_terms,
        n,
        q,
        host_challenges,
        n_prover_challenges,
        num_rounds,
        host_claim0_out,
        host_round_evals_out,
        timings_out);
}

extern "C" const char* sc_last_error_message(void) {
    return g_last_error.c_str();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sumcheck_terms_u32_cuda", &torch_sumcheck_terms_u32_cuda,
          "Uploaded-base u32 SumCheck over arbitrary product-term expressions");
}
