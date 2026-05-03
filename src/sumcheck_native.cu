#include "sumcheck_native.h"

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <chrono>
#include <cstdint>

// ============================================================================
// Experimental Montgomery multiply path for q = 2^32 - 5.
// This is a correctness-first normal-domain wrapper:
//
//   normal a,b -> Montgomery aR,bR -> MontMul -> normal ab
//
// This is expected to be slower than a fully Montgomery-domain SumCheck,
// but it lets us test Montgomery reduction correctness/performance in isolation.
// ============================================================================

#ifndef SUMCHECK_EXPERIMENTAL_MONTGOMERY_MUL
#define SUMCHECK_EXPERIMENTAL_MONTGOMERY_MUL 1
#endif

__device__ __forceinline__ uint32_t sumcheck_mont_redc_q32(uint64_t t) {
    constexpr uint64_t Q = 4294967291ULL;
    constexpr uint32_t NPRIME = 0xCCCCCCCDu; // -q^{-1} mod 2^32

    uint32_t m = static_cast<uint32_t>(t) * NPRIME;
    uint64_t mn = static_cast<uint64_t>(m) * Q;

    // REDC needs (t + m*q) >> 32.  The sum can require 65 bits,
    // so preserve the carry from the 64-bit addition.
    uint64_t sum = t + mn;
    uint64_t carry = (sum < t) ? 1ULL : 0ULL;
    uint64_t u = (sum >> 32) | (carry << 32);

    if (u >= Q) {
        u -= Q;
    }
    return static_cast<uint32_t>(u);
}

__device__ __forceinline__ uint32_t sumcheck_mont_to_mont_q32(uint32_t a) {
    // R mod q = 2^32 mod (2^32 - 5) = 5.
    uint64_t z = static_cast<uint64_t>(a) * 5ULL;

    // z is small, but use correction instead of %.
    constexpr uint64_t Q = 4294967291ULL;
    while (z >= Q) {
        z -= Q;
    }
    return static_cast<uint32_t>(z);
}

__device__ __forceinline__ uint32_t sumcheck_mont_mul_mont_q32(uint32_t a_mont, uint32_t b_mont) {
    return sumcheck_mont_redc_q32(static_cast<uint64_t>(a_mont) * static_cast<uint64_t>(b_mont));
}

__device__ __forceinline__ uint32_t sumcheck_mont_from_mont_q32(uint32_t a_mont) {
    return sumcheck_mont_redc_q32(static_cast<uint64_t>(a_mont));
}

__device__ __forceinline__ uint32_t sumcheck_mont_mul_normal_q32(uint32_t a, uint32_t b) {
    uint32_t a_mont = sumcheck_mont_to_mont_q32(a);
    uint32_t b_mont = sumcheck_mont_to_mont_q32(b);
    uint32_t c_mont = sumcheck_mont_mul_mont_q32(a_mont, b_mont);
    return sumcheck_mont_from_mont_q32(c_mont);
}

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




// ============================================================================
// Specialized HyperPlonk-style u32 SumCheck fast paths.
// These are fixed-template kernels used to compare against the generic
// term_offsets / term_vars backend.
// ============================================================================

namespace hp_spec {

using u32 = uint32_t;
using u64 = uint64_t;

constexpr int THREADS = 128;
constexpr int MAX_BLOCKS = 4096;
constexpr u32 Q32_FAST = 4294967291u; // 2^32 - 5

__device__ __forceinline__ u32 add_mod(u32 a, u32 b, u32 q) {
    u64 s = static_cast<u64>(a) + static_cast<u64>(b);
    if (s >= q) s -= q;
    if (s >= q) s %= q;
    return static_cast<u32>(s);
}

__device__ __forceinline__ u32 sub_mod(u32 a, u32 b, u32 q) {
    return (a >= b) ? static_cast<u32>(a - b) : static_cast<u32>(static_cast<u64>(a) + q - b);
}

__device__ __forceinline__ u32 reduce_q32_fast(u64 z) {
    // For q = 2^32 - 5, use 2^32 == 5 mod q.
    u64 r = (z & 0xffffffffULL) + 5ULL * (z >> 32);
    r = (r & 0xffffffffULL) + 5ULL * (r >> 32);
    r = (r & 0xffffffffULL) + 5ULL * (r >> 32);

    while (r >= Q32_FAST) r -= Q32_FAST;
    return static_cast<u32>(r);
}

__device__ __forceinline__ u32 mul_mod(u32 a, u32 b, u32 q) {
    u64 z = static_cast<u64>(a) * static_cast<u64>(b);
    if (q == Q32_FAST) {
#if SUMCHECK_EXPERIMENTAL_MONTGOMERY_MUL
        return sumcheck_mont_mul_normal_q32(a, b);
#else
        return reduce_q32_fast(z);
#endif
    }
    return static_cast<u32>(z % static_cast<u64>(q));
}

__device__ __forceinline__ u32 line_eval(u32 z, u32 o, u32 t, u32 q) {
    u32 delta = sub_mod(o, z, q);
    return add_mod(z, mul_mod(delta, t, q), q);
}

__device__ __forceinline__ u32 load_line(
    const u32* __restrict__ tables,
    int len,
    int row,
    int idx,
    int half,
    u32 x,
    u32 q) {
    const u32* r = tables + static_cast<size_t>(row) * static_cast<size_t>(len);
    return line_eval(r[idx], r[idx + half], x, q);
}

__device__ __forceinline__ u32 prod2(u32 a, u32 b, u32 q) {
    return mul_mod(a, b, q);
}

__device__ __forceinline__ u32 prod3(u32 a, u32 b, u32 c, u32 q) {
    return mul_mod(mul_mod(a, b, q), c, q);
}

__device__ __forceinline__ u32 prod4(u32 a, u32 b, u32 c, u32 d, u32 q) {
    return mul_mod(mul_mod(mul_mod(a, b, q), c, q), d, q);
}

__device__ __forceinline__ u32 prod5(u32 a, u32 b, u32 c, u32 d, u32 e, u32 q) {
    return mul_mod(mul_mod(mul_mod(mul_mod(a, b, q), c, q), d, q), e, q);
}

__device__ __forceinline__ u32 prod6(u32 a, u32 b, u32 c, u32 d, u32 e, u32 f, u32 q) {
    return mul_mod(prod5(a, b, c, d, e, q), f, q);
}

__device__ __forceinline__ u32 prod7(u32 a, u32 b, u32 c, u32 d, u32 e, u32 f, u32 g, u32 q) {
    return mul_mod(prod6(a, b, c, d, e, f, q), g, q);
}

__device__ __forceinline__ u32 eval_poly_at_x(
    const u32* __restrict__ tables,
    int len,
    int idx,
    int half,
    int poly_id,
    u32 x,
    u32 q) {

    u32 acc = 0;

    if (poly_id == 0) {
        // vanilla_gate:
        // qL*w1 + qR*w2 + qM*w1*w2 + neg_qO*w3 + qC
        u32 qL = load_line(tables, len, 0, idx, half, x, q);
        u32 w1 = load_line(tables, len, 1, idx, half, x, q);
        u32 qR = load_line(tables, len, 2, idx, half, x, q);
        u32 w2 = load_line(tables, len, 3, idx, half, x, q);
        u32 qM = load_line(tables, len, 4, idx, half, x, q);
        u32 nqO = load_line(tables, len, 5, idx, half, x, q);
        u32 w3 = load_line(tables, len, 6, idx, half, x, q);
        u32 qC = load_line(tables, len, 7, idx, half, x, q);

        acc = add_mod(acc, prod2(qL, w1, q), q);
        acc = add_mod(acc, prod2(qR, w2, q), q);
        acc = add_mod(acc, prod3(qM, w1, w2, q), q);
        acc = add_mod(acc, prod2(nqO, w3, q), q);
        acc = add_mod(acc, qC, q);
        return acc;
    }

    if (poly_id == 1) {
        // vanilla_zero = vanilla_gate * fr, termwise.
        u32 qL = load_line(tables, len, 0, idx, half, x, q);
        u32 w1 = load_line(tables, len, 1, idx, half, x, q);
        u32 qR = load_line(tables, len, 2, idx, half, x, q);
        u32 w2 = load_line(tables, len, 3, idx, half, x, q);
        u32 qM = load_line(tables, len, 4, idx, half, x, q);
        u32 nqO = load_line(tables, len, 5, idx, half, x, q);
        u32 w3 = load_line(tables, len, 6, idx, half, x, q);
        u32 qC = load_line(tables, len, 7, idx, half, x, q);
        u32 fr = load_line(tables, len, 8, idx, half, x, q);

        acc = add_mod(acc, prod3(qL, w1, fr, q), q);
        acc = add_mod(acc, prod3(qR, w2, fr, q), q);
        acc = add_mod(acc, prod4(qM, w1, w2, fr, q), q);
        acc = add_mod(acc, prod3(nqO, w3, fr, q), q);
        acc = add_mod(acc, prod2(qC, fr, q), q);
        return acc;
    }

    if (poly_id == 2) {
        // vanilla_perm:
        // (pi - p1*p2 + alpha*phi*D1*D2*D3 - alpha*N1*N2*N3) * fr
        // Signs/scalars are folded into rows.
        u32 pi = load_line(tables, len, 0, idx, half, x, q);
        u32 np1 = load_line(tables, len, 1, idx, half, x, q);
        u32 p2 = load_line(tables, len, 2, idx, half, x, q);
        u32 aphi = load_line(tables, len, 3, idx, half, x, q);
        u32 D1 = load_line(tables, len, 4, idx, half, x, q);
        u32 D2 = load_line(tables, len, 5, idx, half, x, q);
        u32 D3 = load_line(tables, len, 6, idx, half, x, q);
        u32 nalphaN1 = load_line(tables, len, 7, idx, half, x, q);
        u32 N2 = load_line(tables, len, 8, idx, half, x, q);
        u32 N3 = load_line(tables, len, 9, idx, half, x, q);
        u32 fr = load_line(tables, len, 10, idx, half, x, q);

        acc = add_mod(acc, prod2(pi, fr, q), q);
        acc = add_mod(acc, prod3(np1, p2, fr, q), q);
        acc = add_mod(acc, prod5(aphi, D1, D2, D3, fr, q), q);
        acc = add_mod(acc, prod4(nalphaN1, N2, N3, fr, q), q);
        return acc;
    }

    if (poly_id == 3) {
        // opencheck_6: y1*k1 + ... + y6*k6, coefficients folded into rows.
        #pragma unroll
        for (int r = 0; r < 6; ++r) {
            acc = add_mod(acc, load_line(tables, len, r, idx, half, x, q), q);
        }
        return acc;
    }

    if (poly_id == 4) {
        // jellyfish_zero structural template.
        u32 q1 = load_line(tables, len, 0, idx, half, x, q);
        u32 w1 = load_line(tables, len, 1, idx, half, x, q);
        u32 q2 = load_line(tables, len, 2, idx, half, x, q);
        u32 w2 = load_line(tables, len, 3, idx, half, x, q);
        u32 q3 = load_line(tables, len, 4, idx, half, x, q);
        u32 w3 = load_line(tables, len, 5, idx, half, x, q);
        u32 q4 = load_line(tables, len, 6, idx, half, x, q);
        u32 w4 = load_line(tables, len, 7, idx, half, x, q);
        u32 qM1 = load_line(tables, len, 8, idx, half, x, q);
        u32 qM2 = load_line(tables, len, 9, idx, half, x, q);
        u32 qH1 = load_line(tables, len, 10, idx, half, x, q);
        u32 qH2 = load_line(tables, len, 11, idx, half, x, q);
        u32 qH3 = load_line(tables, len, 12, idx, half, x, q);
        u32 qH4 = load_line(tables, len, 13, idx, half, x, q);
        u32 nqO = load_line(tables, len, 14, idx, half, x, q);
        u32 w5 = load_line(tables, len, 15, idx, half, x, q);
        u32 qECC = load_line(tables, len, 16, idx, half, x, q);
        u32 fr = load_line(tables, len, 17, idx, half, x, q);
        u32 qC = load_line(tables, len, 18, idx, half, x, q);

        acc = add_mod(acc, prod3(q1, w1, fr, q), q);
        acc = add_mod(acc, prod3(q2, w2, fr, q), q);
        acc = add_mod(acc, prod3(q3, w3, fr, q), q);
        acc = add_mod(acc, prod3(q4, w4, fr, q), q);
        acc = add_mod(acc, prod4(qM1, w1, w2, fr, q), q);
        acc = add_mod(acc, prod4(qM2, w3, w4, fr, q), q);

        u32 w1_5 = prod5(w1, w1, w1, w1, w1, q);
        u32 w2_5 = prod5(w2, w2, w2, w2, w2, q);
        u32 w3_5 = prod5(w3, w3, w3, w3, w3, q);
        u32 w4_5 = prod5(w4, w4, w4, w4, w4, q);

        acc = add_mod(acc, prod3(qH1, w1_5, fr, q), q);
        acc = add_mod(acc, prod3(qH2, w2_5, fr, q), q);
        acc = add_mod(acc, prod3(qH3, w3_5, fr, q), q);
        acc = add_mod(acc, prod3(qH4, w4_5, fr, q), q);
        acc = add_mod(acc, prod3(nqO, w5, fr, q), q);
        acc = add_mod(acc, prod6(qECC, w1, w2, w3, w4, fr, q), q);
        acc = add_mod(acc, prod2(qC, fr, q), q);

        return acc;
    }

    if (poly_id == 8) {
        // baseline_linear: a
        u32 a = load_line(tables, len, 0, idx, half, x, q);
        return a;
    }

    if (poly_id == 9) {
        // baseline_mul: a*b
        u32 a = load_line(tables, len, 0, idx, half, x, q);
        u32 b = load_line(tables, len, 1, idx, half, x, q);
        return prod2(a, b, q);
    }

    if (poly_id == 10) {
        // baseline_mul_add: a*b + c
        u32 a = load_line(tables, len, 0, idx, half, x, q);
        u32 b = load_line(tables, len, 1, idx, half, x, q);
        u32 c = load_line(tables, len, 2, idx, half, x, q);
        return add_mod(prod2(a, b, q), c, q);
    }

    if (poly_id == 11) {
        // baseline_cubic_product: a*b*c
        u32 a = load_line(tables, len, 0, idx, half, x, q);
        u32 b = load_line(tables, len, 1, idx, half, x, q);
        u32 c = load_line(tables, len, 2, idx, half, x, q);
        return prod3(a, b, c, q);
    }

    if (poly_id == 5 || poly_id == 6 || poly_id == 7) {
        // custom gate family:
        //   q1*w1 + q2*w2 + qH*w1^k*w2 + qC
        //
        // poly_id 5: degree 3 term qH*w1*w2
        // poly_id 6: degree 5 term qH*w1^3*w2
        // poly_id 7: degree 7 term qH*w1^5*w2
        u32 q1 = load_line(tables, len, 0, idx, half, x, q);
        u32 w1 = load_line(tables, len, 1, idx, half, x, q);
        u32 q2 = load_line(tables, len, 2, idx, half, x, q);
        u32 w2 = load_line(tables, len, 3, idx, half, x, q);
        u32 qH = load_line(tables, len, 4, idx, half, x, q);
        u32 qC = load_line(tables, len, 5, idx, half, x, q);

        acc = add_mod(acc, prod2(q1, w1, q), q);
        acc = add_mod(acc, prod2(q2, w2, q), q);

        if (poly_id == 5) {
            acc = add_mod(acc, prod3(qH, w1, w2, q), q);
        } else if (poly_id == 6) {
            acc = add_mod(acc, prod5(qH, w1, w1, w1, w2, q), q);
        } else {
            acc = add_mod(acc, prod7(qH, w1, w1, w1, w1, w1, w2, q), q);
        }

        acc = add_mod(acc, qC, q);
        return acc;
    }

    return 0;
}

__device__ __forceinline__ u32 warp_reduce_add_mod(u32 value, u32 q) {
    // Reduce one u32 modular sum within a warp using register shuffles.
    // This avoids the full shared-memory tree reduction used by the first
    // specialized implementation.
    unsigned mask = 0xffffffffu;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        u32 other = __shfl_down_sync(mask, value, offset);
        value = add_mod(value, other, q);
    }

    return value;
}

__global__ void eval_kernel(
    const u32* __restrict__ tables,
    int len,
    int poly_id,
    int degree,
    u32 q,
    u32* __restrict__ partials) {

    extern __shared__ u32 shared[];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = (blockDim.x + 31) >> 5;
    const int half = len >> 1;

    u32 local[8];

    #pragma unroll
    for (int x = 0; x < 8; ++x) {
        local[x] = 0;
    }

    // Each thread accumulates local partial sums for all evaluation points.
    for (int i = blockIdx.x * blockDim.x + tid;
         i < half;
         i += blockDim.x * gridDim.x) {
        for (int x = 0; x <= degree; ++x) {
            u32 y = eval_poly_at_x(
                tables,
                len,
                i,
                half,
                poly_id,
                static_cast<u32>(x),
                q
            );
            local[x] = add_mod(local[x], y, q);
        }
    }

    // First reduce inside each warp using shuffle instructions.
    for (int x = 0; x <= degree; ++x) {
        u32 reduced = warp_reduce_add_mod(local[x], q);
        if (lane == 0) {
            shared[x * num_warps + warp_id] = reduced;
        }
    }

    __syncthreads();

    // Then let the first warp reduce the per-warp partials.
    if (warp_id == 0) {
        for (int x = 0; x <= degree; ++x) {
            u32 value = 0;

            if (lane < num_warps) {
                value = shared[x * num_warps + lane];
            }

            value = warp_reduce_add_mod(value, q);

            if (lane == 0) {
                partials[
                    static_cast<size_t>(blockIdx.x) *
                    static_cast<size_t>(degree + 1) +
                    x
                ] = value;
            }
        }
    }
}

__global__ void reduce_kernel(
    const u32* __restrict__ partials,
    int num_blocks,
    int degree,
    u32 q,
    u32* __restrict__ out_round) {

    __shared__ u32 scratch[THREADS];
    int x = blockIdx.x;
    int tid = threadIdx.x;

    u32 acc = 0;
    for (int b = tid; b < num_blocks; b += blockDim.x) {
        acc = add_mod(acc, partials[static_cast<size_t>(b) * static_cast<size_t>(degree + 1) + x], q);
    }

    scratch[tid] = acc;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = add_mod(scratch[tid], scratch[tid + stride], q);
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_round[x] = scratch[0];
    }
}

__global__ void update_kernel(
    const u32* __restrict__ in_tables,
    int rows,
    int len,
    const u32* __restrict__ challenges,
    int round,
    u32 q,
    u32* __restrict__ out_tables) {

    int half = len >> 1;
    size_t total = static_cast<size_t>(rows) * static_cast<size_t>(half);

    for (size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<size_t>(gridDim.x) * blockDim.x) {

        int row = static_cast<int>(linear / half);
        int i = static_cast<int>(linear - static_cast<size_t>(row) * static_cast<size_t>(half));

        const u32* in_row = in_tables + static_cast<size_t>(row) * static_cast<size_t>(len);
        u32* out_row = out_tables + static_cast<size_t>(row) * static_cast<size_t>(half);

        u32 r = challenges[round];
        out_row[i] = line_eval(in_row[i], in_row[i + half], r, q);
    }
}

int degree_for_poly(int poly_id) {
    switch (poly_id) {
        case 0: return 3; // vanilla_gate
        case 1: return 4; // vanilla_zero
        case 2: return 5; // vanilla_perm
        case 3: return 1; // opencheck_6
        case 4: return 7; // jellyfish_zero
        case 5: return 3; // custom_gate_deg3
        case 6: return 5; // custom_gate_deg5
        case 7: return 7; // custom_gate_deg7
        case 8: return 1; // baseline_linear: a
        case 9: return 2; // baseline_mul: a*b
        case 10: return 2; // baseline_mul_add: a*b+c
        case 11: return 3; // baseline_cubic_product: a*b*c
        default: throw std::invalid_argument("unknown HyperPlonk poly_id");
    }
}

int rows_for_poly(int poly_id) {
    switch (poly_id) {
        case 0: return 8;
        case 1: return 9;
        case 2: return 11;
        case 3: return 6;
        case 4: return 19;
        case 5: return 6;
        case 6: return 6;
        case 7: return 6;
        case 8: return 1;
        case 9: return 2;
        case 10: return 3;
        case 11: return 3;
        default: throw std::invalid_argument("unknown HyperPlonk poly_id");
    }
}

bool is_power_of_two_i64(int64_t x) {
    return x > 0 && ((x & (x - 1)) == 0);
}

int log2_exact_i64(int64_t x) {
    int r = 0;
    while (x > 1) {
        x >>= 1;
        ++r;
    }
    return r;
}

void check_last_cuda(const char* label) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(err));
    }
}

} // namespace hp_spec

torch::Tensor sumcheck_hyperplonk_u32_cuda(
    torch::Tensor eval_tables,
    torch::Tensor challenges,
    uint64_t modulus,
    int64_t poly_id_64) {

    using namespace hp_spec;

    int poly_id = static_cast<int>(poly_id_64);
    int degree = degree_for_poly(poly_id);
    int rows = rows_for_poly(poly_id);

    if (!eval_tables.is_cuda()) {
        throw std::invalid_argument("eval_tables must be CUDA");
    }
    if (eval_tables.scalar_type() != torch::kUInt32) {
        throw std::invalid_argument("eval_tables must be torch.uint32");
    }
    if (eval_tables.dim() != 2) {
        throw std::invalid_argument("eval_tables must have shape (rows, N)");
    }
    if (eval_tables.size(0) < rows) {
        throw std::invalid_argument("eval_tables has too few rows for requested poly_id");
    }
    if (!is_power_of_two_i64(eval_tables.size(1))) {
        throw std::invalid_argument("N must be a power of two");
    }
    if (modulus > 0xffffffffULL) {
        throw std::invalid_argument("sumcheck_hyperplonk_u32_cuda requires a u32 modulus");
    }

    const c10::cuda::CUDAGuard device_guard(eval_tables.device());

    auto tables = eval_tables.narrow(0, 0, rows).contiguous();
    auto rs = challenges.to(eval_tables.options().dtype(torch::kUInt32)).contiguous();

    int initial_len = static_cast<int>(tables.size(1));
    int rounds = log2_exact_i64(initial_len);

    if (rs.dim() != 1 || rs.size(0) < rounds) {
        throw std::invalid_argument("challenges must have shape at least (log2(N),)");
    }

    auto output = torch::empty({rounds, degree + 1}, tables.options());
    auto current = tables;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    u32 q = static_cast<u32>(modulus);

    for (int round = 0; round < rounds; ++round) {
        int len = static_cast<int>(current.size(1));
        int half = len >> 1;
        int blocks = std::min(MAX_BLOCKS, std::max(1, (half + THREADS - 1) / THREADS));

        auto partials = torch::empty({blocks, degree + 1}, tables.options());
        size_t shmem = static_cast<size_t>(degree + 1) * THREADS * sizeof(u32);

        eval_kernel<<<blocks, THREADS, shmem, stream>>>(
            reinterpret_cast<const u32*>(current.data_ptr<uint32_t>()),
            len,
            poly_id,
            degree,
            q,
            reinterpret_cast<u32*>(partials.data_ptr<uint32_t>()));
        check_last_cuda("hp_spec eval_kernel");

        reduce_kernel<<<degree + 1, THREADS, 0, stream>>>(
            reinterpret_cast<const u32*>(partials.data_ptr<uint32_t>()),
            blocks,
            degree,
            q,
            reinterpret_cast<u32*>(output[round].data_ptr<uint32_t>()));
        check_last_cuda("hp_spec reduce_kernel");

        if (round + 1 < rounds) {
            auto next = torch::empty({rows, half}, tables.options());
            int upd_blocks = std::min(
                MAX_BLOCKS,
                std::max(1, (rows * half + THREADS - 1) / THREADS));

            update_kernel<<<upd_blocks, THREADS, 0, stream>>>(
                reinterpret_cast<const u32*>(current.data_ptr<uint32_t>()),
                rows,
                len,
                reinterpret_cast<const u32*>(rs.data_ptr<uint32_t>()),
                round,
                q,
                reinterpret_cast<u32*>(next.data_ptr<uint32_t>()));
            check_last_cuda("hp_spec update_kernel");

            current = next;
        }
    }

    return output;
}




// ============================================================================
// Full Montgomery-domain specialized SumCheck experiment.
// Converts input tables/challenges to Montgomery form once, computes all rounds
// in Montgomery form, then converts the final round-evaluation matrix back.
// ============================================================================

namespace hp_full_mont {

using u32 = uint32_t;
using u64 = uint64_t;

constexpr int THREADS = 128;
constexpr int MAX_BLOCKS = 4096;

__device__ __forceinline__ u32 add_mod(u32 a, u32 b, u32 q) {
    u64 s = static_cast<u64>(a) + static_cast<u64>(b);
    if (s >= q) s -= q;
    if (s >= q) s %= q;
    return static_cast<u32>(s);
}

__device__ __forceinline__ u32 sub_mod(u32 a, u32 b, u32 q) {
    return (a >= b) ? static_cast<u32>(a - b)
                    : static_cast<u32>(static_cast<u64>(a) + q - b);
}

__device__ __forceinline__ u32 mont_to_mont(u32 a) {
    return sumcheck_mont_to_mont_q32(a);
}

__device__ __forceinline__ u32 mont_from_mont(u32 a_mont) {
    return sumcheck_mont_from_mont_q32(a_mont);
}

__device__ __forceinline__ u32 mont_mul(u32 a_mont, u32 b_mont) {
    return sumcheck_mont_mul_mont_q32(a_mont, b_mont);
}

__device__ __forceinline__ u32 line_eval_mont(u32 z_mont, u32 o_mont, u32 t_mont, u32 q) {
    u32 delta = sub_mod(o_mont, z_mont, q);
    return add_mod(z_mont, mont_mul(t_mont, delta), q);
}

__device__ __forceinline__ u32 load_line_mont(
    const u32* __restrict__ tables,
    int len,
    int row,
    int idx,
    int half,
    u32 t_mont,
    u32 q) {

    const u32* r = tables + static_cast<size_t>(row) * static_cast<size_t>(len);
    return line_eval_mont(r[idx], r[idx + half], t_mont, q);
}

__device__ __forceinline__ u32 prod2(u32 a, u32 b) {
    return mont_mul(a, b);
}

__device__ __forceinline__ u32 prod3(u32 a, u32 b, u32 c) {
    return mont_mul(mont_mul(a, b), c);
}

__device__ __forceinline__ u32 prod4(u32 a, u32 b, u32 c, u32 d) {
    return mont_mul(mont_mul(mont_mul(a, b), c), d);
}

__device__ __forceinline__ u32 prod5(u32 a, u32 b, u32 c, u32 d, u32 e) {
    return mont_mul(mont_mul(mont_mul(mont_mul(a, b), c), d), e);
}

__device__ __forceinline__ u32 prod7(u32 a, u32 b, u32 c, u32 d, u32 e, u32 f, u32 g) {
    return mont_mul(mont_mul(prod5(a, b, c, d, e), f), g);
}

__device__ __forceinline__ u32 eval_poly_at_x_mont(
    const u32* __restrict__ tables,
    int len,
    int idx,
    int half,
    int poly_id,
    u32 x_mont,
    u32 q) {

    u32 acc = 0;

    if (poly_id == 8) {
        return load_line_mont(tables, len, 0, idx, half, x_mont, q);
    }

    if (poly_id == 9) {
        u32 a = load_line_mont(tables, len, 0, idx, half, x_mont, q);
        u32 b = load_line_mont(tables, len, 1, idx, half, x_mont, q);
        return prod2(a, b);
    }

    if (poly_id == 10) {
        u32 a = load_line_mont(tables, len, 0, idx, half, x_mont, q);
        u32 b = load_line_mont(tables, len, 1, idx, half, x_mont, q);
        u32 c = load_line_mont(tables, len, 2, idx, half, x_mont, q);
        return add_mod(prod2(a, b), c, q);
    }

    if (poly_id == 11) {
        u32 a = load_line_mont(tables, len, 0, idx, half, x_mont, q);
        u32 b = load_line_mont(tables, len, 1, idx, half, x_mont, q);
        u32 c = load_line_mont(tables, len, 2, idx, half, x_mont, q);
        return prod3(a, b, c);
    }

    if (poly_id == 0) {
        u32 qL  = load_line_mont(tables, len, 0, idx, half, x_mont, q);
        u32 w1  = load_line_mont(tables, len, 1, idx, half, x_mont, q);
        u32 qR  = load_line_mont(tables, len, 2, idx, half, x_mont, q);
        u32 w2  = load_line_mont(tables, len, 3, idx, half, x_mont, q);
        u32 qM  = load_line_mont(tables, len, 4, idx, half, x_mont, q);
        u32 nqO = load_line_mont(tables, len, 5, idx, half, x_mont, q);
        u32 w3  = load_line_mont(tables, len, 6, idx, half, x_mont, q);
        u32 qC  = load_line_mont(tables, len, 7, idx, half, x_mont, q);

        acc = add_mod(acc, prod2(qL, w1), q);
        acc = add_mod(acc, prod2(qR, w2), q);
        acc = add_mod(acc, prod3(qM, w1, w2), q);
        acc = add_mod(acc, prod2(nqO, w3), q);
        acc = add_mod(acc, qC, q);
        return acc;
    }

    if (poly_id == 1) {
        u32 qL  = load_line_mont(tables, len, 0, idx, half, x_mont, q);
        u32 w1  = load_line_mont(tables, len, 1, idx, half, x_mont, q);
        u32 qR  = load_line_mont(tables, len, 2, idx, half, x_mont, q);
        u32 w2  = load_line_mont(tables, len, 3, idx, half, x_mont, q);
        u32 qM  = load_line_mont(tables, len, 4, idx, half, x_mont, q);
        u32 nqO = load_line_mont(tables, len, 5, idx, half, x_mont, q);
        u32 w3  = load_line_mont(tables, len, 6, idx, half, x_mont, q);
        u32 qC  = load_line_mont(tables, len, 7, idx, half, x_mont, q);
        u32 fr  = load_line_mont(tables, len, 8, idx, half, x_mont, q);

        acc = add_mod(acc, prod3(qL, w1, fr), q);
        acc = add_mod(acc, prod3(qR, w2, fr), q);
        acc = add_mod(acc, prod4(qM, w1, w2, fr), q);
        acc = add_mod(acc, prod3(nqO, w3, fr), q);
        acc = add_mod(acc, prod2(qC, fr), q);
        return acc;
    }

    if (poly_id == 2) {
        u32 pi   = load_line_mont(tables, len, 0, idx, half, x_mont, q);
        u32 np1  = load_line_mont(tables, len, 1, idx, half, x_mont, q);
        u32 p2   = load_line_mont(tables, len, 2, idx, half, x_mont, q);
        u32 aphi = load_line_mont(tables, len, 3, idx, half, x_mont, q);
        u32 D1   = load_line_mont(tables, len, 4, idx, half, x_mont, q);
        u32 D2   = load_line_mont(tables, len, 5, idx, half, x_mont, q);
        u32 D3   = load_line_mont(tables, len, 6, idx, half, x_mont, q);
        u32 nN1  = load_line_mont(tables, len, 7, idx, half, x_mont, q);
        u32 N2   = load_line_mont(tables, len, 8, idx, half, x_mont, q);
        u32 N3   = load_line_mont(tables, len, 9, idx, half, x_mont, q);
        u32 fr   = load_line_mont(tables, len, 10, idx, half, x_mont, q);

        acc = add_mod(acc, prod2(pi, fr), q);
        acc = add_mod(acc, prod3(np1, p2, fr), q);
        acc = add_mod(acc, prod5(aphi, D1, D2, D3, fr), q);
        acc = add_mod(acc, prod4(nN1, N2, N3, fr), q);
        return acc;
    }

    if (poly_id == 3) {
        #pragma unroll
        for (int r = 0; r < 6; ++r) {
            acc = add_mod(acc, load_line_mont(tables, len, r, idx, half, x_mont, q), q);
        }
        return acc;
    }

    if (poly_id == 5 || poly_id == 6 || poly_id == 7) {
        u32 q1 = load_line_mont(tables, len, 0, idx, half, x_mont, q);
        u32 w1 = load_line_mont(tables, len, 1, idx, half, x_mont, q);
        u32 q2 = load_line_mont(tables, len, 2, idx, half, x_mont, q);
        u32 w2 = load_line_mont(tables, len, 3, idx, half, x_mont, q);
        u32 qH = load_line_mont(tables, len, 4, idx, half, x_mont, q);
        u32 qC = load_line_mont(tables, len, 5, idx, half, x_mont, q);

        acc = add_mod(acc, prod2(q1, w1), q);
        acc = add_mod(acc, prod2(q2, w2), q);

        if (poly_id == 5) {
            acc = add_mod(acc, prod3(qH, w1, w2), q);
        } else if (poly_id == 6) {
            acc = add_mod(acc, prod5(qH, w1, w1, w1, w2), q);
        } else {
            acc = add_mod(acc, prod7(qH, w1, w1, w1, w1, w1, w2), q);
        }

        acc = add_mod(acc, qC, q);
        return acc;
    }

    return 0;
}

__device__ __forceinline__ u32 warp_reduce_add_mod(u32 value, u32 q) {
    unsigned mask = 0xffffffffu;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        u32 other = __shfl_down_sync(mask, value, offset);
        value = add_mod(value, other, q);
    }

    return value;
}

__global__ void convert_to_mont_kernel(
    const u32* __restrict__ in,
    size_t n,
    u32* __restrict__ out) {

    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        out[i] = mont_to_mont(in[i]);
    }
}

__global__ void convert_from_mont_kernel(
    const u32* __restrict__ in,
    size_t n,
    u32* __restrict__ out) {

    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        out[i] = mont_from_mont(in[i]);
    }
}

__global__ void eval_kernel(
    const u32* __restrict__ tables,
    int len,
    int poly_id,
    int degree,
    u32 q,
    u32* __restrict__ partials) {

    extern __shared__ u32 shared[];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = (blockDim.x + 31) >> 5;
    const int half = len >> 1;

    u32 local[8];

    #pragma unroll
    for (int x = 0; x < 8; ++x) {
        local[x] = 0;
    }

    for (int i = blockIdx.x * blockDim.x + tid;
         i < half;
         i += blockDim.x * gridDim.x) {
        for (int x = 0; x <= degree; ++x) {
            u32 x_mont = mont_to_mont(static_cast<u32>(x));
            u32 y = eval_poly_at_x_mont(
                tables,
                len,
                i,
                half,
                poly_id,
                x_mont,
                q
            );
            local[x] = add_mod(local[x], y, q);
        }
    }

    for (int x = 0; x <= degree; ++x) {
        u32 reduced = warp_reduce_add_mod(local[x], q);
        if (lane == 0) {
            shared[x * num_warps + warp_id] = reduced;
        }
    }

    __syncthreads();

    if (warp_id == 0) {
        for (int x = 0; x <= degree; ++x) {
            u32 value = 0;

            if (lane < num_warps) {
                value = shared[x * num_warps + lane];
            }

            value = warp_reduce_add_mod(value, q);

            if (lane == 0) {
                partials[
                    static_cast<size_t>(blockIdx.x) *
                    static_cast<size_t>(degree + 1) +
                    x
                ] = value;
            }
        }
    }
}

__global__ void reduce_kernel(
    const u32* __restrict__ partials,
    int num_blocks,
    int degree,
    u32 q,
    u32* __restrict__ out_round) {

    __shared__ u32 scratch[THREADS];
    int x = blockIdx.x;
    int tid = threadIdx.x;

    u32 acc = 0;
    for (int b = tid; b < num_blocks; b += blockDim.x) {
        acc = add_mod(acc, partials[static_cast<size_t>(b) * static_cast<size_t>(degree + 1) + x], q);
    }

    scratch[tid] = acc;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = add_mod(scratch[tid], scratch[tid + stride], q);
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_round[x] = scratch[0];
    }
}

__global__ void update_kernel(
    const u32* __restrict__ in_tables,
    int rows,
    int len,
    const u32* __restrict__ challenges_mont,
    int round,
    u32 q,
    u32* __restrict__ out_tables) {

    int half = len >> 1;
    size_t total = static_cast<size_t>(rows) * static_cast<size_t>(half);

    for (size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<size_t>(gridDim.x) * blockDim.x) {

        int row = static_cast<int>(linear / half);
        int i = static_cast<int>(linear - static_cast<size_t>(row) * static_cast<size_t>(half));

        const u32* in_row = in_tables + static_cast<size_t>(row) * static_cast<size_t>(len);
        u32* out_row = out_tables + static_cast<size_t>(row) * static_cast<size_t>(half);

        u32 r_mont = challenges_mont[round];
        out_row[i] = line_eval_mont(in_row[i], in_row[i + half], r_mont, q);
    }
}

} // namespace hp_full_mont

torch::Tensor sumcheck_hyperplonk_full_mont_u32_cuda(
    torch::Tensor eval_tables,
    torch::Tensor challenges,
    uint64_t modulus,
    int64_t poly_id_64) {

    using namespace hp_full_mont;

    int poly_id = static_cast<int>(poly_id_64);

    if (poly_id == 4) {
        throw std::invalid_argument("poly_id 4 is unused and not supported by full Montgomery path");
    }

    int degree = hp_spec::degree_for_poly(poly_id);
    int rows = hp_spec::rows_for_poly(poly_id);

    if (!eval_tables.is_cuda()) {
        throw std::invalid_argument("eval_tables must be CUDA");
    }
    if (eval_tables.scalar_type() != torch::kUInt32) {
        throw std::invalid_argument("eval_tables must be torch.uint32");
    }
    if (eval_tables.dim() != 2) {
        throw std::invalid_argument("eval_tables must have shape (rows, N)");
    }
    if (eval_tables.size(0) < rows) {
        throw std::invalid_argument("eval_tables has too few rows for requested poly_id");
    }
    if (!hp_spec::is_power_of_two_i64(eval_tables.size(1))) {
        throw std::invalid_argument("N must be a power of two");
    }
    if (modulus != 4294967291ULL) {
        throw std::invalid_argument("full Montgomery experiment currently requires q = 4294967291");
    }

    const c10::cuda::CUDAGuard device_guard(eval_tables.device());

    auto normal_tables = eval_tables.narrow(0, 0, rows).contiguous();
    auto normal_chals = challenges.to(eval_tables.options().dtype(torch::kUInt32)).contiguous();

    int initial_len = static_cast<int>(normal_tables.size(1));
    int rounds = hp_spec::log2_exact_i64(initial_len);

    if (normal_chals.dim() != 1 || normal_chals.size(0) < rounds) {
        throw std::invalid_argument("challenges must have shape at least (log2(N),)");
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    u32 q = static_cast<u32>(modulus);

    auto current = torch::empty_like(normal_tables);
    auto chals_mont = torch::empty_like(normal_chals);

    int conv_threads = THREADS;
    int conv_blocks_tables = std::min(
        MAX_BLOCKS,
        static_cast<int>(
            std::max<int64_t>(1, (normal_tables.numel() + conv_threads - 1) / conv_threads)
        )
    );
    int conv_blocks_chals = std::min(
        MAX_BLOCKS,
        static_cast<int>(
            std::max<int64_t>(1, (normal_chals.numel() + conv_threads - 1) / conv_threads)
        )
    );

    convert_to_mont_kernel<<<conv_blocks_tables, conv_threads, 0, stream>>>(
        reinterpret_cast<const u32*>(normal_tables.data_ptr<uint32_t>()),
        static_cast<size_t>(normal_tables.numel()),
        reinterpret_cast<u32*>(current.data_ptr<uint32_t>()));
    hp_spec::check_last_cuda("full_mont convert tables");

    convert_to_mont_kernel<<<conv_blocks_chals, conv_threads, 0, stream>>>(
        reinterpret_cast<const u32*>(normal_chals.data_ptr<uint32_t>()),
        static_cast<size_t>(normal_chals.numel()),
        reinterpret_cast<u32*>(chals_mont.data_ptr<uint32_t>()));
    hp_spec::check_last_cuda("full_mont convert challenges");

    auto output_mont = torch::empty({rounds, degree + 1}, normal_tables.options());

    for (int round = 0; round < rounds; ++round) {
        int len = static_cast<int>(current.size(1));
        int half = len >> 1;
        int blocks = std::min(MAX_BLOCKS, std::max(1, (half + THREADS - 1) / THREADS));

        auto partials = torch::empty({blocks, degree + 1}, normal_tables.options());
        size_t shmem = static_cast<size_t>(degree + 1) * THREADS * sizeof(u32);

        eval_kernel<<<blocks, THREADS, shmem, stream>>>(
            reinterpret_cast<const u32*>(current.data_ptr<uint32_t>()),
            len,
            poly_id,
            degree,
            q,
            reinterpret_cast<u32*>(partials.data_ptr<uint32_t>()));
        hp_spec::check_last_cuda("full_mont eval_kernel");

        reduce_kernel<<<degree + 1, THREADS, 0, stream>>>(
            reinterpret_cast<const u32*>(partials.data_ptr<uint32_t>()),
            blocks,
            degree,
            q,
            reinterpret_cast<u32*>(output_mont[round].data_ptr<uint32_t>()));
        hp_spec::check_last_cuda("full_mont reduce_kernel");

        if (round + 1 < rounds) {
            auto next = torch::empty({rows, half}, normal_tables.options());

            int upd_blocks = std::min(
                MAX_BLOCKS,
                std::max(1, (rows * half + THREADS - 1) / THREADS));

            update_kernel<<<upd_blocks, THREADS, 0, stream>>>(
                reinterpret_cast<const u32*>(current.data_ptr<uint32_t>()),
                rows,
                len,
                reinterpret_cast<const u32*>(chals_mont.data_ptr<uint32_t>()),
                round,
                q,
                reinterpret_cast<u32*>(next.data_ptr<uint32_t>()));
            hp_spec::check_last_cuda("full_mont update_kernel");

            current = next;
        }
    }

    auto output_normal = torch::empty_like(output_mont);
    int conv_blocks_out = std::min(
        MAX_BLOCKS,
        static_cast<int>(
            std::max<int64_t>(1, (output_mont.numel() + conv_threads - 1) / conv_threads)
        )
    );

    convert_from_mont_kernel<<<conv_blocks_out, conv_threads, 0, stream>>>(
        reinterpret_cast<const u32*>(output_mont.data_ptr<uint32_t>()),
        static_cast<size_t>(output_mont.numel()),
        reinterpret_cast<u32*>(output_normal.data_ptr<uint32_t>()));
    hp_spec::check_last_cuda("full_mont convert output");

    return output_normal;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sumcheck_terms_u32_cuda", &torch_sumcheck_terms_u32_cuda,
          "Uploaded-base u32 SumCheck over arbitrary product-term expressions");
    m.def("sumcheck_hyperplonk_u32_cuda",
          &sumcheck_hyperplonk_u32_cuda,
          "Specialized u32 SumCheck for fixed HyperPlonk-style polynomial templates");
    m.def("sumcheck_hyperplonk_full_mont_u32_cuda",
          &sumcheck_hyperplonk_full_mont_u32_cuda,
          "Experimental full Montgomery-domain specialized u32 SumCheck");


}
