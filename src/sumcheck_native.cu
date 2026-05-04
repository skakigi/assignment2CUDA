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
#define SUMCHECK_EXPERIMENTAL_MONTGOMERY_MUL 0
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

#include "cuda/sumcheck_strategy.cuh"


// Known fixed-template polynomial IDs used by HyperPlonk-style benchmarks.
// Keep these IDs stable because Python benchmark scripts pass them through.
#include "cuda/poly_templates.cuh"











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





constexpr int kEvalTStride = 4;
inline size_t eval_shared_bytes_for_variant(
    SpecEvalStrategy variant,
    int eval_threads,
    int32_t n_terms,
    int32_t total_term_vars) {
    const size_t metadata_bytes = static_cast<size_t>(n_terms + 1 + total_term_vars) * sizeof(int32_t);
    if (variant == SpecEvalStrategy::kBaselineShared ||
        variant == SpecEvalStrategy::kMleTiledShared) {
        return static_cast<size_t>(kEvalTStride) * static_cast<size_t>(eval_threads) * sizeof(uint32_t) +
               metadata_bytes;
    }
    const size_t warp_count = static_cast<size_t>((eval_threads + 31) / 32);
    if (variant == SpecEvalStrategy::kMleTiledWarp) {
        return static_cast<size_t>(kEvalTStride) * warp_count * sizeof(uint32_t) + metadata_bytes;
    }
    return static_cast<size_t>(kEvalTStride) * warp_count * sizeof(uint64_t) + metadata_bytes;
}









}  // namespace






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
        return reduce_q32_fast(z);
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

    if (poly_id == kPolyVanillaGate) {
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

    if (poly_id == kPolyVanillaZero) {
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

    if (poly_id == kPolyVanillaPerm) {
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

    if (poly_id == kPolyOpencheck6) {
        // opencheck_6: y1*k1 + ... + y6*k6, coefficients folded into rows.
        #pragma unroll
        for (int r = 0; r < 6; ++r) {
            acc = add_mod(acc, load_line(tables, len, r, idx, half, x, q), q);
        }
        return acc;
    }


    if (poly_id == kPolyBaselineLinear) {
        // baseline_linear: a
        u32 a = load_line(tables, len, 0, idx, half, x, q);
        return a;
    }

    if (poly_id == kPolyBaselineMul) {
        // baseline_mul: a*b
        u32 a = load_line(tables, len, 0, idx, half, x, q);
        u32 b = load_line(tables, len, 1, idx, half, x, q);
        return prod2(a, b, q);
    }

    if (poly_id == kPolyBaselineMulAdd) {
        // baseline_mul_add: a*b + c
        u32 a = load_line(tables, len, 0, idx, half, x, q);
        u32 b = load_line(tables, len, 1, idx, half, x, q);
        u32 c = load_line(tables, len, 2, idx, half, x, q);
        return add_mod(prod2(a, b, q), c, q);
    }

    if (poly_id == kPolyBaselineCubicProduct) {
        // baseline_cubic_product: a*b*c
        u32 a = load_line(tables, len, 0, idx, half, x, q);
        u32 b = load_line(tables, len, 1, idx, half, x, q);
        u32 c = load_line(tables, len, 2, idx, half, x, q);
        return prod3(a, b, c, q);
    }


    if (poly_id == kPolyAdvancedA2B2C) {
        // advanced_a2b2c old hp_spec: a*a*b*b*c
        u32 a = load_line(tables, len, 0, idx, half, x, q);
        u32 b = load_line(tables, len, 1, idx, half, x, q);
        u32 c = load_line(tables, len, 2, idx, half, x, q);
        return prod5(a, a, b, b, c, q);
    }

    if (poly_id == kPolyAdvancedAbcPlusDe) {
        // advanced_abc_plus_de old hp_spec: a*b*c + d*e
        u32 a = load_line(tables, len, 0, idx, half, x, q);
        u32 b = load_line(tables, len, 1, idx, half, x, q);
        u32 c = load_line(tables, len, 2, idx, half, x, q);
        u32 d = load_line(tables, len, 3, idx, half, x, q);
        u32 e = load_line(tables, len, 4, idx, half, x, q);
        return add_mod(prod3(a, b, c, q), prod2(d, e, q), q);
    }

    if (poly_id == kPolyAdvancedAbcgPlusDeg) {
        // advanced_abcg_plus_deg old hp_spec: a*b*c*g + d*e*g
        u32 a = load_line(tables, len, 0, idx, half, x, q);
        u32 b = load_line(tables, len, 1, idx, half, x, q);
        u32 c = load_line(tables, len, 2, idx, half, x, q);
        u32 d = load_line(tables, len, 3, idx, half, x, q);
        u32 e = load_line(tables, len, 4, idx, half, x, q);
        u32 g = load_line(tables, len, 5, idx, half, x, q);
        return add_mod(prod4(a, b, c, g, q), prod3(d, e, g, q), q);
    }

    if (poly_id == kPolyDegreeSweepDeg3 ||
        poly_id == kPolyDegreeSweepDeg5 ||
        poly_id == kPolyDegreeSweepDeg7) {
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

        if (poly_id == kPolyDegreeSweepDeg3) {
            acc = add_mod(acc, prod3(qH, w1, w2, q), q);
        } else if (poly_id == kPolyDegreeSweepDeg5) {
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

    if (poly_id == kPolyBaselineLinear) {
        return load_line_mont(tables, len, 0, idx, half, x_mont, q);
    }

    if (poly_id == kPolyBaselineMul) {
        u32 a = load_line_mont(tables, len, 0, idx, half, x_mont, q);
        u32 b = load_line_mont(tables, len, 1, idx, half, x_mont, q);
        return prod2(a, b);
    }

    if (poly_id == kPolyBaselineMulAdd) {
        u32 a = load_line_mont(tables, len, 0, idx, half, x_mont, q);
        u32 b = load_line_mont(tables, len, 1, idx, half, x_mont, q);
        u32 c = load_line_mont(tables, len, 2, idx, half, x_mont, q);
        return add_mod(prod2(a, b), c, q);
    }

    if (poly_id == kPolyBaselineCubicProduct) {
        u32 a = load_line_mont(tables, len, 0, idx, half, x_mont, q);
        u32 b = load_line_mont(tables, len, 1, idx, half, x_mont, q);
        u32 c = load_line_mont(tables, len, 2, idx, half, x_mont, q);
        return prod3(a, b, c);
    }


    if (poly_id == kPolyAdvancedA2B2C) {
        // advanced_a2b2c u32 full mont: a*a*b*b*c
        u32 a = load_line_mont(tables, len, 0, idx, half, x_mont, q);
        u32 b = load_line_mont(tables, len, 1, idx, half, x_mont, q);
        u32 c = load_line_mont(tables, len, 2, idx, half, x_mont, q);
        return prod5(a, a, b, b, c);
    }

    if (poly_id == kPolyAdvancedAbcPlusDe) {
        // advanced_abc_plus_de u32 full mont: a*b*c + d*e
        u32 a = load_line_mont(tables, len, 0, idx, half, x_mont, q);
        u32 b = load_line_mont(tables, len, 1, idx, half, x_mont, q);
        u32 c = load_line_mont(tables, len, 2, idx, half, x_mont, q);
        u32 d = load_line_mont(tables, len, 3, idx, half, x_mont, q);
        u32 e = load_line_mont(tables, len, 4, idx, half, x_mont, q);
        return add_mod(prod3(a, b, c), prod2(d, e), q);
    }

    if (poly_id == kPolyAdvancedAbcgPlusDeg) {
        // advanced_abcg_plus_deg u32 full mont: a*b*c*g + d*e*g
        u32 a = load_line_mont(tables, len, 0, idx, half, x_mont, q);
        u32 b = load_line_mont(tables, len, 1, idx, half, x_mont, q);
        u32 c = load_line_mont(tables, len, 2, idx, half, x_mont, q);
        u32 d = load_line_mont(tables, len, 3, idx, half, x_mont, q);
        u32 e = load_line_mont(tables, len, 4, idx, half, x_mont, q);
        u32 g = load_line_mont(tables, len, 5, idx, half, x_mont, q);
        return add_mod(prod4(a, b, c, g), prod3(d, e, g), q);
    }

    if (poly_id == kPolyVanillaGate) {
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

    if (poly_id == kPolyVanillaZero) {
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

    if (poly_id == kPolyVanillaPerm) {
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

    if (poly_id == kPolyOpencheck6) {
        #pragma unroll
        for (int r = 0; r < 6; ++r) {
            acc = add_mod(acc, load_line_mont(tables, len, r, idx, half, x_mont, q), q);
        }
        return acc;
    }

    if (poly_id == kPolyDegreeSweepDeg3 ||
        poly_id == kPolyDegreeSweepDeg5 ||
        poly_id == kPolyDegreeSweepDeg7) {
        u32 q1 = load_line_mont(tables, len, 0, idx, half, x_mont, q);
        u32 w1 = load_line_mont(tables, len, 1, idx, half, x_mont, q);
        u32 q2 = load_line_mont(tables, len, 2, idx, half, x_mont, q);
        u32 w2 = load_line_mont(tables, len, 3, idx, half, x_mont, q);
        u32 qH = load_line_mont(tables, len, 4, idx, half, x_mont, q);
        u32 qC = load_line_mont(tables, len, 5, idx, half, x_mont, q);

        acc = add_mod(acc, prod2(q1, w1), q);
        acc = add_mod(acc, prod2(q2, w2), q);

        if (poly_id == kPolyDegreeSweepDeg3) {
            acc = add_mod(acc, prod3(qH, w1, w2), q);
        } else if (poly_id == kPolyDegreeSweepDeg5) {
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



// ============================================================================
// Full Montgomery-domain generic SumCheck experiment.
// This is the generic term_offsets / term_vars backend, but it converts
// eval_tables and challenges into Montgomery form once, runs all rounds in
// Montgomery form, then converts the output matrix back once.
// ============================================================================

namespace hp_full_mont {

__device__ __forceinline__ u32 eval_terms_at_x_mont(
    const u32* __restrict__ tables,
    int len,
    int idx,
    int half,
    const int32_t* __restrict__ term_offsets,
    const int32_t* __restrict__ term_vars,
    int num_terms,
    u32 x_mont,
    u32 q) {

    // Montgomery representation of one:
    // R mod q = 2^32 mod (2^32 - 5) = 5.
    constexpr u32 MONT_ONE = 5u;

    u32 acc = 0;

    for (int term = 0; term < num_terms; ++term) {
        int start = term_offsets[term];
        int end = term_offsets[term + 1];

        u32 prod = MONT_ONE;

        for (int j = start; j < end; ++j) {
            int row = term_vars[j];
            u32 value = load_line_mont(
                tables,
                len,
                row,
                idx,
                half,
                x_mont,
                q
            );
            prod = mont_mul(prod, value);
        }

        acc = add_mod(acc, prod, q);
    }

    return acc;
}

__global__ void eval_terms_kernel(
    const u32* __restrict__ tables,
    int len,
    const int32_t* __restrict__ term_offsets,
    const int32_t* __restrict__ term_vars,
    int num_terms,
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

            u32 y = eval_terms_at_x_mont(
                tables,
                len,
                i,
                half,
                term_offsets,
                term_vars,
                num_terms,
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

__global__ void claim0_from_output_kernel(
    const u32* __restrict__ output_normal,
    int degree,
    u32 q,
    u32* __restrict__ claim0) {

    // SumCheck claim before round 0 is g_0(0) + g_0(1).
    // Degree is at least 1 for all current templates.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (degree >= 1) {
            claim0[0] = add_mod(output_normal[0], output_normal[1], q);
        } else {
            claim0[0] = output_normal[0];
        }
    }
}

} // namespace hp_full_mont

std::tuple<torch::Tensor, torch::Tensor> sumcheck_terms_full_mont_u32_cuda(
    torch::Tensor eval_tables,
    torch::Tensor challenges,
    torch::Tensor term_offsets,
    torch::Tensor term_vars,
    uint64_t modulus) {

    using namespace hp_full_mont;

    if (!eval_tables.is_cuda()) {
        throw std::invalid_argument("eval_tables must be CUDA");
    }
    if (eval_tables.scalar_type() != torch::kUInt32) {
        throw std::invalid_argument("eval_tables must be torch.uint32");
    }
    if (eval_tables.dim() != 2) {
        throw std::invalid_argument("eval_tables must have shape (rows, N)");
    }
    if (!hp_spec::is_power_of_two_i64(eval_tables.size(1))) {
        throw std::invalid_argument("N must be a power of two");
    }
    if (modulus != 4294967291ULL) {
        throw std::invalid_argument("full Montgomery generic experiment currently requires q = 4294967291");
    }

    const c10::cuda::CUDAGuard device_guard(eval_tables.device());

    auto offsets_cpu = term_offsets.to(torch::kCPU).to(torch::kInt32).contiguous();
    auto vars_cpu = term_vars.to(torch::kCPU).to(torch::kInt32).contiguous();

    if (offsets_cpu.dim() != 1 || offsets_cpu.size(0) < 2) {
        throw std::invalid_argument("term_offsets must be a 1D tensor with at least 2 entries");
    }
    if (vars_cpu.dim() != 1) {
        throw std::invalid_argument("term_vars must be a 1D tensor");
    }

    int num_terms = static_cast<int>(offsets_cpu.size(0)) - 1;
    const int32_t* offsets_ptr = offsets_cpu.data_ptr<int32_t>();
    const int32_t* vars_ptr = vars_cpu.data_ptr<int32_t>();

    int degree = 0;
    int max_row = -1;

    for (int t = 0; t < num_terms; ++t) {
        int start = offsets_ptr[t];
        int end = offsets_ptr[t + 1];

        if (start < 0 || end < start || end > vars_cpu.size(0)) {
            throw std::invalid_argument("invalid term_offsets");
        }

        int term_degree = end - start;
        degree = std::max(degree, term_degree);

        for (int j = start; j < end; ++j) {
            int row = vars_ptr[j];
            if (row < 0) {
                throw std::invalid_argument("term_vars contains a negative row index");
            }
            max_row = std::max(max_row, row);
        }
    }

    if (degree > 7) {
        throw std::invalid_argument("full Montgomery generic path currently supports degree <= 7");
    }
    if (max_row >= eval_tables.size(0)) {
        throw std::invalid_argument("term_vars references a row outside eval_tables");
    }

    int rows = static_cast<int>(eval_tables.size(0));

    auto normal_tables = eval_tables.contiguous();
    auto normal_chals = challenges.to(eval_tables.options().dtype(torch::kUInt32)).contiguous();

    int initial_len = static_cast<int>(normal_tables.size(1));
    int rounds = hp_spec::log2_exact_i64(initial_len);

    if (normal_chals.dim() != 1 || normal_chals.size(0) < rounds) {
        throw std::invalid_argument("challenges must have shape at least (log2(N),)");
    }

    auto int_opts_dev = torch::TensorOptions()
        .device(eval_tables.device())
        .dtype(torch::kInt32);

    auto offsets_dev = term_offsets.to(int_opts_dev).contiguous();
    auto vars_dev = term_vars.to(int_opts_dev).contiguous();

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
    hp_spec::check_last_cuda("terms_full_mont convert tables");

    convert_to_mont_kernel<<<conv_blocks_chals, conv_threads, 0, stream>>>(
        reinterpret_cast<const u32*>(normal_chals.data_ptr<uint32_t>()),
        static_cast<size_t>(normal_chals.numel()),
        reinterpret_cast<u32*>(chals_mont.data_ptr<uint32_t>()));
    hp_spec::check_last_cuda("terms_full_mont convert challenges");

    auto output_mont = torch::empty({rounds, degree + 1}, normal_tables.options());

    for (int round = 0; round < rounds; ++round) {
        int len = static_cast<int>(current.size(1));
        int half = len >> 1;
        int blocks = std::min(MAX_BLOCKS, std::max(1, (half + THREADS - 1) / THREADS));

        auto partials = torch::empty({blocks, degree + 1}, normal_tables.options());
        size_t shmem = static_cast<size_t>(degree + 1) * THREADS * sizeof(u32);

        eval_terms_kernel<<<blocks, THREADS, shmem, stream>>>(
            reinterpret_cast<const u32*>(current.data_ptr<uint32_t>()),
            len,
            reinterpret_cast<const int32_t*>(offsets_dev.data_ptr<int32_t>()),
            reinterpret_cast<const int32_t*>(vars_dev.data_ptr<int32_t>()),
            num_terms,
            degree,
            q,
            reinterpret_cast<u32*>(partials.data_ptr<uint32_t>()));
        hp_spec::check_last_cuda("terms_full_mont eval_terms_kernel");

        reduce_kernel<<<degree + 1, THREADS, 0, stream>>>(
            reinterpret_cast<const u32*>(partials.data_ptr<uint32_t>()),
            blocks,
            degree,
            q,
            reinterpret_cast<u32*>(output_mont[round].data_ptr<uint32_t>()));
        hp_spec::check_last_cuda("terms_full_mont reduce_kernel");

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
            hp_spec::check_last_cuda("terms_full_mont update_kernel");

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
    hp_spec::check_last_cuda("terms_full_mont convert output");

    auto claim0 = torch::empty({}, normal_tables.options());

    claim0_from_output_kernel<<<1, 1, 0, stream>>>(
        reinterpret_cast<const u32*>(output_normal.data_ptr<uint32_t>()),
        degree,
        q,
        reinterpret_cast<u32*>(claim0.data_ptr<uint32_t>()));
    hp_spec::check_last_cuda("terms_full_mont claim0");

    return std::make_tuple(claim0, output_normal);
}



// ============================================================================
// u64 Montgomery arithmetic experiment.
// Field: q = 2^64 - 2^32 + 1 = 0xffffffff00000001
//
// This block only adds a multiplication smoke-test entrypoint first:
//   montgomery_u64_mul_test_cuda(a, b)
//
// It computes (a*b mod q) using full Montgomery conversion:
//   normal -> Montgomery -> Montgomery multiply -> normal.
// ============================================================================

namespace mont_u64_exp {

using u64 = uint64_t;

constexpr u64 Q64 = 0xffffffff00000001ULL;
constexpr u64 NPRIME = 0xfffffffeffffffffULL;   // -q^{-1} mod 2^64
constexpr u64 R2_MOD_Q = 0xfffffffe00000001ULL; // R^2 mod q
constexpr u64 R_MINUS_Q = 0x00000000ffffffffULL;

__device__ __forceinline__ u64 add_mod(u64 a, u64 b) {
    u64 s = a + b;
    bool carry = s < a;

    if (carry) {
        // a+b = 2^64 + s. Since q = 2^64 - R_MINUS_Q,
        // subtracting q is equivalent to s + R_MINUS_Q.
        s += R_MINUS_Q;
    } else if (s >= Q64) {
        s -= Q64;
    }

    if (s >= Q64) {
        s -= Q64;
    }
    return s;
}

__device__ __forceinline__ u64 sub_mod(u64 a, u64 b) {
    return (a >= b) ? (a - b) : (a + (Q64 - b));
}

__device__ __forceinline__ u64 redc_from_128(u64 hi, u64 lo) {
    // Montgomery REDC for t = hi*2^64 + lo.
    //
    // m = lo * (-q^{-1}) mod 2^64
    // u = (t + m*q) / 2^64
    // if u >= q: u -= q
    //
    // The high-word sum can overflow by one bit because q is close to 2^64.
    // If that happens, the 65-bit u definitely exceeds q, so subtracting q
    // is equivalent to adding (2^64 - q) = R_MINUS_Q to the wrapped low word.

    u64 m = lo * NPRIME;

    u64 mq_lo = m * Q64;
    u64 mq_hi = __umul64hi(m, Q64);

    u64 sum_lo = lo + mq_lo;
    u64 carry0 = (sum_lo < lo) ? 1ULL : 0ULL;

    u64 u = hi + mq_hi;
    bool carry1 = (u < hi);

    u64 u2 = u + carry0;
    bool carry2 = carry1 || (u2 < u);
    u = u2;

    if (carry2) {
        u += R_MINUS_Q;
    } else if (u >= Q64) {
        u -= Q64;
    }

    if (u >= Q64) {
        u -= Q64;
    }

    return u;
}

__device__ __forceinline__ u64 mont_mul(u64 a_mont, u64 b_mont) {
    u64 lo = a_mont * b_mont;
    u64 hi = __umul64hi(a_mont, b_mont);
    return redc_from_128(hi, lo);
}

__device__ __forceinline__ u64 to_mont(u64 a) {
    // aR mod q = MontMul(a, R^2 mod q)
    return mont_mul(a, R2_MOD_Q);
}

__device__ __forceinline__ u64 from_mont(u64 a_mont) {
    // a = MontMul(aR, 1)
    return redc_from_128(0, a_mont);
}

__device__ __forceinline__ u64 mul_normal(u64 a, u64 b) {
    u64 am = to_mont(a);
    u64 bm = to_mont(b);
    u64 cm = mont_mul(am, bm);
    return from_mont(cm);
}

__global__ void mul_test_kernel(
    const u64* __restrict__ a,
    const u64* __restrict__ b,
    u64* __restrict__ out,
    size_t n) {

    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        out[i] = mul_normal(a[i], b[i]);
    }
}

} // namespace mont_u64_exp

torch::Tensor montgomery_u64_mul_test_cuda(torch::Tensor a, torch::Tensor b) {
    using namespace mont_u64_exp;

    if (!a.is_cuda() || !b.is_cuda()) {
        throw std::invalid_argument("a and b must be CUDA tensors");
    }
    if (a.scalar_type() != torch::kUInt64 || b.scalar_type() != torch::kUInt64) {
        throw std::invalid_argument("a and b must be torch.uint64");
    }
    if (a.sizes() != b.sizes()) {
        throw std::invalid_argument("a and b must have the same shape");
    }

    const c10::cuda::CUDAGuard device_guard(a.device());

    auto aa = a.contiguous();
    auto bb = b.contiguous();
    auto out = torch::empty_like(aa);

    constexpr int THREADS = 128;
    constexpr int MAX_BLOCKS = 4096;

    size_t n = static_cast<size_t>(aa.numel());
    int blocks = std::min<int>(
        MAX_BLOCKS,
        std::max<int>(1, static_cast<int>((n + THREADS - 1) / THREADS))
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    mul_test_kernel<<<blocks, THREADS, 0, stream>>>(
        reinterpret_cast<const u64*>(aa.data_ptr<uint64_t>()),
        reinterpret_cast<const u64*>(bb.data_ptr<uint64_t>()),
        reinterpret_cast<u64*>(out.data_ptr<uint64_t>()),
        n);

    hp_spec::check_last_cuda("montgomery_u64_mul_test_cuda");

    return out;
}



// ============================================================================
// Generic u64 full Montgomery-domain SumCheck.
// Field: q = 2^64 - 2^32 + 1.
// This backend converts eval_tables/challenges once into Montgomery form,
// executes every SumCheck round in Montgomery form, and converts outputs once.
// ============================================================================

namespace mont_u64_exp {

constexpr u64 MONT_ONE = 0x00000000ffffffffULL; // R mod q for q = 2^64 - 2^32 + 1
constexpr int SUMCHECK_THREADS = 128;
constexpr int SUMCHECK_MAX_BLOCKS = 4096;

__device__ __forceinline__ u64 line_eval_mont(u64 z_mont, u64 o_mont, u64 t_mont) {
    u64 delta = sub_mod(o_mont, z_mont);
    return add_mod(z_mont, mont_mul(t_mont, delta));
}

__device__ __forceinline__ u64 load_line_mont(
    const u64* __restrict__ tables,
    int len,
    int row,
    int idx,
    int half,
    u64 x_mont) {

    const u64* r = tables + static_cast<size_t>(row) * static_cast<size_t>(len);
    return line_eval_mont(r[idx], r[idx + half], x_mont);
}

__device__ __forceinline__ u64 eval_terms_at_x_mont(
    const u64* __restrict__ tables,
    int len,
    int idx,
    int half,
    const int32_t* __restrict__ term_offsets,
    const int32_t* __restrict__ term_vars,
    int num_terms,
    u64 x_mont) {

    u64 acc = 0;

    for (int term = 0; term < num_terms; ++term) {
        int start = term_offsets[term];
        int end = term_offsets[term + 1];

        u64 prod = MONT_ONE;

        for (int j = start; j < end; ++j) {
            int row = term_vars[j];
            u64 value = load_line_mont(tables, len, row, idx, half, x_mont);
            prod = mont_mul(prod, value);
        }

        acc = add_mod(acc, prod);
    }

    return acc;
}

__device__ __forceinline__ u64 warp_reduce_add_mod_u64(u64 value) {
    unsigned mask = 0xffffffffu;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        u64 other = __shfl_down_sync(mask, value, offset);
        value = add_mod(value, other);
    }

    return value;
}

__global__ void convert_to_mont_u64_kernel(
    const u64* __restrict__ in,
    size_t n,
    u64* __restrict__ out) {

    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        out[i] = to_mont(in[i]);
    }
}

__global__ void convert_from_mont_u64_kernel(
    const u64* __restrict__ in,
    size_t n,
    u64* __restrict__ out) {

    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        out[i] = from_mont(in[i]);
    }
}

__global__ void eval_terms_sumcheck_u64_kernel(
    const u64* __restrict__ tables,
    int len,
    const int32_t* __restrict__ term_offsets,
    const int32_t* __restrict__ term_vars,
    int num_terms,
    int degree,
    u64* __restrict__ partials) {

    extern __shared__ u64 shared[];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = (blockDim.x + 31) >> 5;
    const int half = len >> 1;

    u64 local[8];

    #pragma unroll
    for (int x = 0; x < 8; ++x) {
        local[x] = 0;
    }

    for (int i = blockIdx.x * blockDim.x + tid;
         i < half;
         i += blockDim.x * gridDim.x) {

        for (int x = 0; x <= degree; ++x) {
            u64 x_mont = to_mont(static_cast<u64>(x));

            u64 y = eval_terms_at_x_mont(
                tables,
                len,
                i,
                half,
                term_offsets,
                term_vars,
                num_terms,
                x_mont
            );

            local[x] = add_mod(local[x], y);
        }
    }

    for (int x = 0; x <= degree; ++x) {
        u64 reduced = warp_reduce_add_mod_u64(local[x]);
        if (lane == 0) {
            shared[x * num_warps + warp_id] = reduced;
        }
    }

    __syncthreads();

    if (warp_id == 0) {
        for (int x = 0; x <= degree; ++x) {
            u64 value = 0;

            if (lane < num_warps) {
                value = shared[x * num_warps + lane];
            }

            value = warp_reduce_add_mod_u64(value);

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

__global__ void reduce_sumcheck_u64_kernel(
    const u64* __restrict__ partials,
    int num_blocks,
    int degree,
    u64* __restrict__ out_round) {

    __shared__ u64 scratch[SUMCHECK_THREADS];

    int x = blockIdx.x;
    int tid = threadIdx.x;

    u64 acc = 0;

    for (int b = tid; b < num_blocks; b += blockDim.x) {
        acc = add_mod(
            acc,
            partials[
                static_cast<size_t>(b) *
                static_cast<size_t>(degree + 1) +
                x
            ]
        );
    }

    scratch[tid] = acc;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = add_mod(scratch[tid], scratch[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_round[x] = scratch[0];
    }
}

__global__ void update_sumcheck_u64_kernel(
    const u64* __restrict__ in_tables,
    int rows,
    int len,
    const u64* __restrict__ challenges_mont,
    int round,
    u64* __restrict__ out_tables) {

    int half = len >> 1;
    size_t total = static_cast<size_t>(rows) * static_cast<size_t>(half);

    for (size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<size_t>(gridDim.x) * blockDim.x) {

        int row = static_cast<int>(linear / half);
        int i = static_cast<int>(
            linear - static_cast<size_t>(row) * static_cast<size_t>(half)
        );

        const u64* in_row =
            in_tables + static_cast<size_t>(row) * static_cast<size_t>(len);
        u64* out_row =
            out_tables + static_cast<size_t>(row) * static_cast<size_t>(half);

        u64 r_mont = challenges_mont[round];
        out_row[i] = line_eval_mont(in_row[i], in_row[i + half], r_mont);
    }
}

__global__ void claim0_from_output_u64_kernel(
    const u64* __restrict__ output_normal,
    int degree,
    u64* __restrict__ claim0) {

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (degree >= 1) {
            claim0[0] = add_mod(output_normal[0], output_normal[1]);
        } else {
            claim0[0] = output_normal[0];
        }
    }
}

} // namespace mont_u64_exp

std::tuple<torch::Tensor, torch::Tensor> sumcheck_terms_full_mont_u64_cuda(
    torch::Tensor eval_tables,
    torch::Tensor challenges,
    torch::Tensor term_offsets,
    torch::Tensor term_vars,
    uint64_t modulus_hi,
    uint64_t modulus_lo) {

    using namespace mont_u64_exp;

    if (modulus_hi != 0 || modulus_lo != Q64) {
        throw std::invalid_argument(
            "u64 Montgomery SumCheck currently requires q = 2^64 - 2^32 + 1"
        );
    }

    if (!eval_tables.is_cuda()) {
        throw std::invalid_argument("eval_tables must be CUDA");
    }
    if (eval_tables.scalar_type() != torch::kUInt64) {
        throw std::invalid_argument("eval_tables must be torch.uint64");
    }
    if (eval_tables.dim() != 2) {
        throw std::invalid_argument("eval_tables must have shape (rows, N)");
    }
    if (!hp_spec::is_power_of_two_i64(eval_tables.size(1))) {
        throw std::invalid_argument("N must be a power of two");
    }

    const c10::cuda::CUDAGuard device_guard(eval_tables.device());

    auto offsets_cpu = term_offsets.to(torch::kCPU).to(torch::kInt32).contiguous();
    auto vars_cpu = term_vars.to(torch::kCPU).to(torch::kInt32).contiguous();

    if (offsets_cpu.dim() != 1 || offsets_cpu.size(0) < 2) {
        throw std::invalid_argument("term_offsets must be a 1D tensor with at least 2 entries");
    }
    if (vars_cpu.dim() != 1) {
        throw std::invalid_argument("term_vars must be a 1D tensor");
    }

    int num_terms = static_cast<int>(offsets_cpu.size(0)) - 1;
    const int32_t* offsets_ptr = offsets_cpu.data_ptr<int32_t>();
    const int32_t* vars_ptr = vars_cpu.data_ptr<int32_t>();

    int degree = 0;
    int max_row = -1;

    for (int t = 0; t < num_terms; ++t) {
        int start = offsets_ptr[t];
        int end = offsets_ptr[t + 1];

        if (start < 0 || end < start || end > vars_cpu.size(0)) {
            throw std::invalid_argument("invalid term_offsets");
        }

        int term_degree = end - start;
        degree = std::max(degree, term_degree);

        for (int j = start; j < end; ++j) {
            int row = vars_ptr[j];
            if (row < 0) {
                throw std::invalid_argument("term_vars contains a negative row index");
            }
            max_row = std::max(max_row, row);
        }
    }

    if (degree > 7) {
        throw std::invalid_argument("u64 generic path currently supports degree <= 7");
    }
    if (max_row >= eval_tables.size(0)) {
        throw std::invalid_argument("term_vars references a row outside eval_tables");
    }

    int rows = static_cast<int>(eval_tables.size(0));
    int initial_len = static_cast<int>(eval_tables.size(1));
    int rounds = hp_spec::log2_exact_i64(initial_len);

    auto normal_tables = eval_tables.contiguous();
    auto normal_chals =
        challenges.to(eval_tables.options().dtype(torch::kUInt64)).contiguous();

    if (normal_chals.dim() != 1 || normal_chals.size(0) < rounds) {
        throw std::invalid_argument("challenges must have shape at least (log2(N),)");
    }

    auto int_opts_dev =
        torch::TensorOptions().device(eval_tables.device()).dtype(torch::kInt32);

    auto offsets_dev = term_offsets.to(int_opts_dev).contiguous();
    auto vars_dev = term_vars.to(int_opts_dev).contiguous();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto current = torch::empty_like(normal_tables);
    auto chals_mont = torch::empty_like(normal_chals);

    int conv_threads = SUMCHECK_THREADS;

    int conv_blocks_tables = std::min(
        SUMCHECK_MAX_BLOCKS,
        static_cast<int>(
            std::max<int64_t>(
                1,
                (normal_tables.numel() + conv_threads - 1) / conv_threads
            )
        )
    );

    int conv_blocks_chals = std::min(
        SUMCHECK_MAX_BLOCKS,
        static_cast<int>(
            std::max<int64_t>(
                1,
                (normal_chals.numel() + conv_threads - 1) / conv_threads
            )
        )
    );

    convert_to_mont_u64_kernel<<<conv_blocks_tables, conv_threads, 0, stream>>>(
        reinterpret_cast<const u64*>(normal_tables.data_ptr<uint64_t>()),
        static_cast<size_t>(normal_tables.numel()),
        reinterpret_cast<u64*>(current.data_ptr<uint64_t>()));
    hp_spec::check_last_cuda("u64 terms convert tables");

    convert_to_mont_u64_kernel<<<conv_blocks_chals, conv_threads, 0, stream>>>(
        reinterpret_cast<const u64*>(normal_chals.data_ptr<uint64_t>()),
        static_cast<size_t>(normal_chals.numel()),
        reinterpret_cast<u64*>(chals_mont.data_ptr<uint64_t>()));
    hp_spec::check_last_cuda("u64 terms convert challenges");

    auto output_mont =
        torch::empty({rounds, degree + 1}, normal_tables.options());

    for (int round = 0; round < rounds; ++round) {
        int len = static_cast<int>(current.size(1));
        int half = len >> 1;

        int blocks = std::min(
            SUMCHECK_MAX_BLOCKS,
            std::max(1, (half + SUMCHECK_THREADS - 1) / SUMCHECK_THREADS)
        );

        auto partials =
            torch::empty({blocks, degree + 1}, normal_tables.options());

        size_t shmem =
            static_cast<size_t>(degree + 1) *
            SUMCHECK_THREADS *
            sizeof(u64);

        eval_terms_sumcheck_u64_kernel<<<blocks, SUMCHECK_THREADS, shmem, stream>>>(
            reinterpret_cast<const u64*>(current.data_ptr<uint64_t>()),
            len,
            reinterpret_cast<const int32_t*>(offsets_dev.data_ptr<int32_t>()),
            reinterpret_cast<const int32_t*>(vars_dev.data_ptr<int32_t>()),
            num_terms,
            degree,
            reinterpret_cast<u64*>(partials.data_ptr<uint64_t>()));
        hp_spec::check_last_cuda("u64 terms eval");

        reduce_sumcheck_u64_kernel<<<degree + 1, SUMCHECK_THREADS, 0, stream>>>(
            reinterpret_cast<const u64*>(partials.data_ptr<uint64_t>()),
            blocks,
            degree,
            reinterpret_cast<u64*>(output_mont[round].data_ptr<uint64_t>()));
        hp_spec::check_last_cuda("u64 terms reduce");

        if (round + 1 < rounds) {
            auto next = torch::empty({rows, half}, normal_tables.options());

            int upd_blocks = std::min(
                SUMCHECK_MAX_BLOCKS,
                std::max(
                    1,
                    (rows * half + SUMCHECK_THREADS - 1) / SUMCHECK_THREADS
                )
            );

            update_sumcheck_u64_kernel<<<upd_blocks, SUMCHECK_THREADS, 0, stream>>>(
                reinterpret_cast<const u64*>(current.data_ptr<uint64_t>()),
                rows,
                len,
                reinterpret_cast<const u64*>(chals_mont.data_ptr<uint64_t>()),
                round,
                reinterpret_cast<u64*>(next.data_ptr<uint64_t>()));
            hp_spec::check_last_cuda("u64 terms update");

            current = next;
        }
    }

    auto output_normal = torch::empty_like(output_mont);

    int conv_blocks_out = std::min(
        SUMCHECK_MAX_BLOCKS,
        static_cast<int>(
            std::max<int64_t>(
                1,
                (output_mont.numel() + conv_threads - 1) / conv_threads
            )
        )
    );

    convert_from_mont_u64_kernel<<<conv_blocks_out, conv_threads, 0, stream>>>(
        reinterpret_cast<const u64*>(output_mont.data_ptr<uint64_t>()),
        static_cast<size_t>(output_mont.numel()),
        reinterpret_cast<u64*>(output_normal.data_ptr<uint64_t>()));
    hp_spec::check_last_cuda("u64 terms convert output");

    auto claim0 = torch::empty({}, normal_tables.options());

    claim0_from_output_u64_kernel<<<1, 1, 0, stream>>>(
        reinterpret_cast<const u64*>(output_normal.data_ptr<uint64_t>()),
        degree,
        reinterpret_cast<u64*>(claim0.data_ptr<uint64_t>()));
    hp_spec::check_last_cuda("u64 terms claim0");

    return std::make_tuple(claim0, output_normal);
}



// ============================================================================
// Specialized u64 full Montgomery-domain SumCheck.
// Field: q = 2^64 - 2^32 + 1.
// Uses fixed polynomial templates selected by poly_id.
// ============================================================================

namespace mont_u64_exp {

__device__ __forceinline__ u64 prod2_spec(u64 a, u64 b) {
    return mont_mul(a, b);
}

__device__ __forceinline__ u64 prod3_spec(u64 a, u64 b, u64 c) {
    return mont_mul(mont_mul(a, b), c);
}

__device__ __forceinline__ u64 prod4_spec(u64 a, u64 b, u64 c, u64 d) {
    return mont_mul(mont_mul(mont_mul(a, b), c), d);
}

__device__ __forceinline__ u64 prod5_spec(u64 a, u64 b, u64 c, u64 d, u64 e) {
    return mont_mul(mont_mul(mont_mul(mont_mul(a, b), c), d), e);
}

__device__ __forceinline__ u64 prod7_spec(
    u64 a,
    u64 b,
    u64 c,
    u64 d,
    u64 e,
    u64 f,
    u64 g) {
    return mont_mul(mont_mul(prod5_spec(a, b, c, d, e), f), g);
}

__device__ __forceinline__ u64 eval_hyperplonk_poly_at_x_mont_u64(
    const u64* __restrict__ tables,
    int len,
    int idx,
    int half,
    int poly_id,
    u64 x_mont) {

    u64 acc = 0;

    if (poly_id == kPolyBaselineLinear) {
        // baseline_linear: a
        return load_line_mont(tables, len, 0, idx, half, x_mont);
    }

    if (poly_id == kPolyBaselineMul) {
        // baseline_mul: a*b
        u64 a = load_line_mont(tables, len, 0, idx, half, x_mont);
        u64 b = load_line_mont(tables, len, 1, idx, half, x_mont);
        return prod2_spec(a, b);
    }

    if (poly_id == kPolyBaselineMulAdd) {
        // baseline_mul_add: a*b + c
        u64 a = load_line_mont(tables, len, 0, idx, half, x_mont);
        u64 b = load_line_mont(tables, len, 1, idx, half, x_mont);
        u64 c = load_line_mont(tables, len, 2, idx, half, x_mont);
        return add_mod(prod2_spec(a, b), c);
    }

    if (poly_id == kPolyBaselineCubicProduct) {
        // baseline_cubic_product: a*b*c
        u64 a = load_line_mont(tables, len, 0, idx, half, x_mont);
        u64 b = load_line_mont(tables, len, 1, idx, half, x_mont);
        u64 c = load_line_mont(tables, len, 2, idx, half, x_mont);
        return prod3_spec(a, b, c);
    }


    if (poly_id == kPolyAdvancedA2B2C) {
        // advanced_a2b2c u64 full mont: a*a*b*b*c
        u64 a = load_line_mont(tables, len, 0, idx, half, x_mont);
        u64 b = load_line_mont(tables, len, 1, idx, half, x_mont);
        u64 c = load_line_mont(tables, len, 2, idx, half, x_mont);
        return prod5_spec(a, a, b, b, c);
    }

    if (poly_id == kPolyAdvancedAbcPlusDe) {
        // advanced_abc_plus_de u64 full mont: a*b*c + d*e
        u64 a = load_line_mont(tables, len, 0, idx, half, x_mont);
        u64 b = load_line_mont(tables, len, 1, idx, half, x_mont);
        u64 c = load_line_mont(tables, len, 2, idx, half, x_mont);
        u64 d = load_line_mont(tables, len, 3, idx, half, x_mont);
        u64 e = load_line_mont(tables, len, 4, idx, half, x_mont);
        return add_mod(prod3_spec(a, b, c), prod2_spec(d, e));
    }

    if (poly_id == kPolyAdvancedAbcgPlusDeg) {
        // advanced_abcg_plus_deg u64 full mont: a*b*c*g + d*e*g
        u64 a = load_line_mont(tables, len, 0, idx, half, x_mont);
        u64 b = load_line_mont(tables, len, 1, idx, half, x_mont);
        u64 c = load_line_mont(tables, len, 2, idx, half, x_mont);
        u64 d = load_line_mont(tables, len, 3, idx, half, x_mont);
        u64 e = load_line_mont(tables, len, 4, idx, half, x_mont);
        u64 g = load_line_mont(tables, len, 5, idx, half, x_mont);
        return add_mod(prod4_spec(a, b, c, g), prod3_spec(d, e, g));
    }

    if (poly_id == kPolyVanillaGate) {
        // vanilla_gate:
        // qL*w1 + qR*w2 + qM*w1*w2 + neg_qO*w3 + qC
        u64 qL  = load_line_mont(tables, len, 0, idx, half, x_mont);
        u64 w1  = load_line_mont(tables, len, 1, idx, half, x_mont);
        u64 qR  = load_line_mont(tables, len, 2, idx, half, x_mont);
        u64 w2  = load_line_mont(tables, len, 3, idx, half, x_mont);
        u64 qM  = load_line_mont(tables, len, 4, idx, half, x_mont);
        u64 nqO = load_line_mont(tables, len, 5, idx, half, x_mont);
        u64 w3  = load_line_mont(tables, len, 6, idx, half, x_mont);
        u64 qC  = load_line_mont(tables, len, 7, idx, half, x_mont);

        acc = add_mod(acc, prod2_spec(qL, w1));
        acc = add_mod(acc, prod2_spec(qR, w2));
        acc = add_mod(acc, prod3_spec(qM, w1, w2));
        acc = add_mod(acc, prod2_spec(nqO, w3));
        acc = add_mod(acc, qC);
        return acc;
    }

    if (poly_id == kPolyVanillaZero) {
        // vanilla_zero = vanilla_gate * fr, termwise.
        u64 qL  = load_line_mont(tables, len, 0, idx, half, x_mont);
        u64 w1  = load_line_mont(tables, len, 1, idx, half, x_mont);
        u64 qR  = load_line_mont(tables, len, 2, idx, half, x_mont);
        u64 w2  = load_line_mont(tables, len, 3, idx, half, x_mont);
        u64 qM  = load_line_mont(tables, len, 4, idx, half, x_mont);
        u64 nqO = load_line_mont(tables, len, 5, idx, half, x_mont);
        u64 w3  = load_line_mont(tables, len, 6, idx, half, x_mont);
        u64 qC  = load_line_mont(tables, len, 7, idx, half, x_mont);
        u64 fr  = load_line_mont(tables, len, 8, idx, half, x_mont);

        acc = add_mod(acc, prod3_spec(qL, w1, fr));
        acc = add_mod(acc, prod3_spec(qR, w2, fr));
        acc = add_mod(acc, prod4_spec(qM, w1, w2, fr));
        acc = add_mod(acc, prod3_spec(nqO, w3, fr));
        acc = add_mod(acc, prod2_spec(qC, fr));
        return acc;
    }

    if (poly_id == kPolyVanillaPerm) {
        // vanilla_perm:
        // (pi - p1*p2 + alpha_phi*D1*D2*D3 - alpha*N1*N2*N3) * fr
        // Signs/scalars are folded into input rows.
        u64 pi   = load_line_mont(tables, len, 0, idx, half, x_mont);
        u64 np1  = load_line_mont(tables, len, 1, idx, half, x_mont);
        u64 p2   = load_line_mont(tables, len, 2, idx, half, x_mont);
        u64 aphi = load_line_mont(tables, len, 3, idx, half, x_mont);
        u64 D1   = load_line_mont(tables, len, 4, idx, half, x_mont);
        u64 D2   = load_line_mont(tables, len, 5, idx, half, x_mont);
        u64 D3   = load_line_mont(tables, len, 6, idx, half, x_mont);
        u64 nN1  = load_line_mont(tables, len, 7, idx, half, x_mont);
        u64 N2   = load_line_mont(tables, len, 8, idx, half, x_mont);
        u64 N3   = load_line_mont(tables, len, 9, idx, half, x_mont);
        u64 fr   = load_line_mont(tables, len, 10, idx, half, x_mont);

        acc = add_mod(acc, prod2_spec(pi, fr));
        acc = add_mod(acc, prod3_spec(np1, p2, fr));
        acc = add_mod(acc, prod5_spec(aphi, D1, D2, D3, fr));
        acc = add_mod(acc, prod4_spec(nN1, N2, N3, fr));
        return acc;
    }

    if (poly_id == kPolyOpencheck6) {
        // opencheck_6: y1*k1 + ... + y6*k6
        #pragma unroll
        for (int r = 0; r < 6; ++r) {
            acc = add_mod(
                acc,
                load_line_mont(tables, len, r, idx, half, x_mont)
            );
        }
        return acc;
    }

    if (poly_id == kPolyDegreeSweepDeg3 ||
        poly_id == kPolyDegreeSweepDeg5 ||
        poly_id == kPolyDegreeSweepDeg7) {
        // degree_sweep:
        // q1*w1 + q2*w2 + qH*w1^k*w2 + qC
        u64 q1 = load_line_mont(tables, len, 0, idx, half, x_mont);
        u64 w1 = load_line_mont(tables, len, 1, idx, half, x_mont);
        u64 q2 = load_line_mont(tables, len, 2, idx, half, x_mont);
        u64 w2 = load_line_mont(tables, len, 3, idx, half, x_mont);
        u64 qH = load_line_mont(tables, len, 4, idx, half, x_mont);
        u64 qC = load_line_mont(tables, len, 5, idx, half, x_mont);

        acc = add_mod(acc, prod2_spec(q1, w1));
        acc = add_mod(acc, prod2_spec(q2, w2));

        if (poly_id == kPolyDegreeSweepDeg3) {
            acc = add_mod(acc, prod3_spec(qH, w1, w2));
        } else if (poly_id == kPolyDegreeSweepDeg5) {
            acc = add_mod(acc, prod5_spec(qH, w1, w1, w1, w2));
        } else {
            acc = add_mod(acc, prod7_spec(qH, w1, w1, w1, w1, w1, w2));
        }

        acc = add_mod(acc, qC);
        return acc;
    }

    return 0;
}

__global__ void eval_hyperplonk_sumcheck_u64_kernel(
    const u64* __restrict__ tables,
    int len,
    int poly_id,
    int degree,
    u64* __restrict__ partials) {

    extern __shared__ u64 shared[];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = (blockDim.x + 31) >> 5;
    const int half = len >> 1;

    u64 local[8];

    #pragma unroll
    for (int x = 0; x < 8; ++x) {
        local[x] = 0;
    }

    for (int i = blockIdx.x * blockDim.x + tid;
         i < half;
         i += blockDim.x * gridDim.x) {

        for (int x = 0; x <= degree; ++x) {
            u64 x_mont = to_mont(static_cast<u64>(x));

            u64 y = eval_hyperplonk_poly_at_x_mont_u64(
                tables,
                len,
                i,
                half,
                poly_id,
                x_mont
            );

            local[x] = add_mod(local[x], y);
        }
    }

    for (int x = 0; x <= degree; ++x) {
        u64 reduced = warp_reduce_add_mod_u64(local[x]);
        if (lane == 0) {
            shared[x * num_warps + warp_id] = reduced;
        }
    }

    __syncthreads();

    if (warp_id == 0) {
        for (int x = 0; x <= degree; ++x) {
            u64 value = 0;

            if (lane < num_warps) {
                value = shared[x * num_warps + lane];
            }

            value = warp_reduce_add_mod_u64(value);

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

} // namespace mont_u64_exp

torch::Tensor sumcheck_hyperplonk_full_mont_u64_cuda(
    torch::Tensor eval_tables,
    torch::Tensor challenges,
    uint64_t modulus_hi,
    uint64_t modulus_lo,
    int64_t poly_id_64) {

    using namespace mont_u64_exp;

    if (modulus_hi != 0 || modulus_lo != Q64) {
        throw std::invalid_argument(
            "u64 Montgomery specialized SumCheck currently requires q = 2^64 - 2^32 + 1"
        );
    }

    int poly_id = static_cast<int>(poly_id_64);


    int degree = degree_for_poly(poly_id);
    int rows = rows_for_poly(poly_id);

    if (!eval_tables.is_cuda()) {
        throw std::invalid_argument("eval_tables must be CUDA");
    }
    if (eval_tables.scalar_type() != torch::kUInt64) {
        throw std::invalid_argument("eval_tables must be torch.uint64");
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

    const c10::cuda::CUDAGuard device_guard(eval_tables.device());

    auto normal_tables = eval_tables.narrow(0, 0, rows).contiguous();
    auto normal_chals =
        challenges.to(eval_tables.options().dtype(torch::kUInt64)).contiguous();

    int initial_len = static_cast<int>(normal_tables.size(1));
    int rounds = hp_spec::log2_exact_i64(initial_len);

    if (normal_chals.dim() != 1 || normal_chals.size(0) < rounds) {
        throw std::invalid_argument("challenges must have shape at least (log2(N),)");
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto current = torch::empty_like(normal_tables);
    auto chals_mont = torch::empty_like(normal_chals);

    int conv_threads = SUMCHECK_THREADS;

    int conv_blocks_tables = std::min(
        SUMCHECK_MAX_BLOCKS,
        static_cast<int>(
            std::max<int64_t>(
                1,
                (normal_tables.numel() + conv_threads - 1) / conv_threads
            )
        )
    );

    int conv_blocks_chals = std::min(
        SUMCHECK_MAX_BLOCKS,
        static_cast<int>(
            std::max<int64_t>(
                1,
                (normal_chals.numel() + conv_threads - 1) / conv_threads
            )
        )
    );

    convert_to_mont_u64_kernel<<<conv_blocks_tables, conv_threads, 0, stream>>>(
        reinterpret_cast<const u64*>(normal_tables.data_ptr<uint64_t>()),
        static_cast<size_t>(normal_tables.numel()),
        reinterpret_cast<u64*>(current.data_ptr<uint64_t>()));
    hp_spec::check_last_cuda("u64 spec convert tables");

    convert_to_mont_u64_kernel<<<conv_blocks_chals, conv_threads, 0, stream>>>(
        reinterpret_cast<const u64*>(normal_chals.data_ptr<uint64_t>()),
        static_cast<size_t>(normal_chals.numel()),
        reinterpret_cast<u64*>(chals_mont.data_ptr<uint64_t>()));
    hp_spec::check_last_cuda("u64 spec convert challenges");

    auto output_mont =
        torch::empty({rounds, degree + 1}, normal_tables.options());

    for (int round = 0; round < rounds; ++round) {
        int len = static_cast<int>(current.size(1));
        int half = len >> 1;

        int blocks = std::min(
            SUMCHECK_MAX_BLOCKS,
            std::max(1, (half + SUMCHECK_THREADS - 1) / SUMCHECK_THREADS)
        );

        auto partials =
            torch::empty({blocks, degree + 1}, normal_tables.options());

        size_t shmem =
            static_cast<size_t>(degree + 1) *
            SUMCHECK_THREADS *
            sizeof(u64);

        eval_hyperplonk_sumcheck_u64_kernel<<<blocks, SUMCHECK_THREADS, shmem, stream>>>(
            reinterpret_cast<const u64*>(current.data_ptr<uint64_t>()),
            len,
            poly_id,
            degree,
            reinterpret_cast<u64*>(partials.data_ptr<uint64_t>()));
        hp_spec::check_last_cuda("u64 spec eval");

        reduce_sumcheck_u64_kernel<<<degree + 1, SUMCHECK_THREADS, 0, stream>>>(
            reinterpret_cast<const u64*>(partials.data_ptr<uint64_t>()),
            blocks,
            degree,
            reinterpret_cast<u64*>(output_mont[round].data_ptr<uint64_t>()));
        hp_spec::check_last_cuda("u64 spec reduce");

        if (round + 1 < rounds) {
            auto next = torch::empty({rows, half}, normal_tables.options());

            int upd_blocks = std::min(
                SUMCHECK_MAX_BLOCKS,
                std::max(
                    1,
                    (rows * half + SUMCHECK_THREADS - 1) / SUMCHECK_THREADS
                )
            );

            update_sumcheck_u64_kernel<<<upd_blocks, SUMCHECK_THREADS, 0, stream>>>(
                reinterpret_cast<const u64*>(current.data_ptr<uint64_t>()),
                rows,
                len,
                reinterpret_cast<const u64*>(chals_mont.data_ptr<uint64_t>()),
                round,
                reinterpret_cast<u64*>(next.data_ptr<uint64_t>()));
            hp_spec::check_last_cuda("u64 spec update");

            current = next;
        }
    }

    auto output_normal = torch::empty_like(output_mont);

    int conv_blocks_out = std::min(
        SUMCHECK_MAX_BLOCKS,
        static_cast<int>(
            std::max<int64_t>(
                1,
                (output_mont.numel() + conv_threads - 1) / conv_threads
            )
        )
    );

    convert_from_mont_u64_kernel<<<conv_blocks_out, conv_threads, 0, stream>>>(
        reinterpret_cast<const u64*>(output_mont.data_ptr<uint64_t>()),
        static_cast<size_t>(output_mont.numel()),
        reinterpret_cast<u64*>(output_normal.data_ptr<uint64_t>()));
    hp_spec::check_last_cuda("u64 spec convert output");

    return output_normal;
}



// ============================================================================
// Hashed-challenge support: u64 one-round generic full-Montgomery SumCheck.
// These entrypoints allow a Python transcript driver to:
//   1. evaluate one SumCheck round,
//   2. hash the round evaluations into a challenge,
//   3. fold the table with that challenge,
//   4. repeat.
// ============================================================================

torch::Tensor sumcheck_terms_full_mont_u64_round_eval_cuda(
    torch::Tensor eval_tables,
    torch::Tensor term_offsets,
    torch::Tensor term_vars,
    uint64_t modulus_hi,
    uint64_t modulus_lo) {

    using namespace mont_u64_exp;

    if (modulus_hi != 0 || modulus_lo != Q64) {
        throw std::invalid_argument(
            "u64 round eval currently requires q = 2^64 - 2^32 + 1"
        );
    }

    if (!eval_tables.is_cuda()) {
        throw std::invalid_argument("eval_tables must be CUDA");
    }
    if (eval_tables.scalar_type() != torch::kUInt64) {
        throw std::invalid_argument("eval_tables must be torch.uint64");
    }
    if (eval_tables.dim() != 2) {
        throw std::invalid_argument("eval_tables must have shape (rows, N)");
    }
    if (!hp_spec::is_power_of_two_i64(eval_tables.size(1))) {
        throw std::invalid_argument("N must be a power of two");
    }
    if (eval_tables.size(1) < 2) {
        throw std::invalid_argument("N must be at least 2 for a SumCheck round");
    }

    const c10::cuda::CUDAGuard device_guard(eval_tables.device());

    auto offsets_cpu = term_offsets.to(torch::kCPU).to(torch::kInt32).contiguous();
    auto vars_cpu = term_vars.to(torch::kCPU).to(torch::kInt32).contiguous();

    if (offsets_cpu.dim() != 1 || offsets_cpu.size(0) < 2) {
        throw std::invalid_argument("term_offsets must be a 1D tensor with at least 2 entries");
    }
    if (vars_cpu.dim() != 1) {
        throw std::invalid_argument("term_vars must be a 1D tensor");
    }

    int num_terms = static_cast<int>(offsets_cpu.size(0)) - 1;
    const int32_t* offsets_ptr = offsets_cpu.data_ptr<int32_t>();
    const int32_t* vars_ptr = vars_cpu.data_ptr<int32_t>();

    int degree = 0;
    int max_row = -1;

    for (int t = 0; t < num_terms; ++t) {
        int start = offsets_ptr[t];
        int end = offsets_ptr[t + 1];

        if (start < 0 || end < start || end > vars_cpu.size(0)) {
            throw std::invalid_argument("invalid term_offsets");
        }

        int term_degree = end - start;
        degree = std::max(degree, term_degree);

        for (int j = start; j < end; ++j) {
            int row = vars_ptr[j];
            if (row < 0) {
                throw std::invalid_argument("term_vars contains a negative row index");
            }
            max_row = std::max(max_row, row);
        }
    }

    if (degree > 7) {
        throw std::invalid_argument("u64 round eval currently supports degree <= 7");
    }
    if (max_row >= eval_tables.size(0)) {
        throw std::invalid_argument("term_vars references a row outside eval_tables");
    }

    auto normal_tables = eval_tables.contiguous();

    auto int_opts_dev =
        torch::TensorOptions().device(eval_tables.device()).dtype(torch::kInt32);

    auto offsets_dev = term_offsets.to(int_opts_dev).contiguous();
    auto vars_dev = term_vars.to(int_opts_dev).contiguous();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int len = static_cast<int>(normal_tables.size(1));
    int half = len >> 1;

    auto current_mont = torch::empty_like(normal_tables);

    int conv_threads = SUMCHECK_THREADS;
    int conv_blocks_tables = std::min(
        SUMCHECK_MAX_BLOCKS,
        static_cast<int>(
            std::max<int64_t>(
                1,
                (normal_tables.numel() + conv_threads - 1) / conv_threads
            )
        )
    );

    convert_to_mont_u64_kernel<<<conv_blocks_tables, conv_threads, 0, stream>>>(
        reinterpret_cast<const u64*>(normal_tables.data_ptr<uint64_t>()),
        static_cast<size_t>(normal_tables.numel()),
        reinterpret_cast<u64*>(current_mont.data_ptr<uint64_t>()));
    hp_spec::check_last_cuda("hashed u64 round convert tables");

    int blocks = std::min(
        SUMCHECK_MAX_BLOCKS,
        std::max(1, (half + SUMCHECK_THREADS - 1) / SUMCHECK_THREADS)
    );

    auto partials = torch::empty({blocks, degree + 1}, normal_tables.options());
    auto output_mont = torch::empty({degree + 1}, normal_tables.options());

    size_t shmem =
        static_cast<size_t>(degree + 1) *
        SUMCHECK_THREADS *
        sizeof(u64);

    eval_terms_sumcheck_u64_kernel<<<blocks, SUMCHECK_THREADS, shmem, stream>>>(
        reinterpret_cast<const u64*>(current_mont.data_ptr<uint64_t>()),
        len,
        reinterpret_cast<const int32_t*>(offsets_dev.data_ptr<int32_t>()),
        reinterpret_cast<const int32_t*>(vars_dev.data_ptr<int32_t>()),
        num_terms,
        degree,
        reinterpret_cast<u64*>(partials.data_ptr<uint64_t>()));
    hp_spec::check_last_cuda("hashed u64 round eval");

    reduce_sumcheck_u64_kernel<<<degree + 1, SUMCHECK_THREADS, 0, stream>>>(
        reinterpret_cast<const u64*>(partials.data_ptr<uint64_t>()),
        blocks,
        degree,
        reinterpret_cast<u64*>(output_mont.data_ptr<uint64_t>()));
    hp_spec::check_last_cuda("hashed u64 round reduce");

    auto output_normal = torch::empty_like(output_mont);

    int conv_blocks_out = std::min(
        SUMCHECK_MAX_BLOCKS,
        static_cast<int>(
            std::max<int64_t>(
                1,
                (output_mont.numel() + conv_threads - 1) / conv_threads
            )
        )
    );

    convert_from_mont_u64_kernel<<<conv_blocks_out, conv_threads, 0, stream>>>(
        reinterpret_cast<const u64*>(output_mont.data_ptr<uint64_t>()),
        static_cast<size_t>(output_mont.numel()),
        reinterpret_cast<u64*>(output_normal.data_ptr<uint64_t>()));
    hp_spec::check_last_cuda("hashed u64 round convert output");

    return output_normal;
}

torch::Tensor fold_full_mont_u64_cuda(
    torch::Tensor eval_tables,
    torch::Tensor challenge,
    uint64_t modulus_hi,
    uint64_t modulus_lo) {

    using namespace mont_u64_exp;

    if (modulus_hi != 0 || modulus_lo != Q64) {
        throw std::invalid_argument(
            "u64 fold currently requires q = 2^64 - 2^32 + 1"
        );
    }

    if (!eval_tables.is_cuda()) {
        throw std::invalid_argument("eval_tables must be CUDA");
    }
    if (eval_tables.scalar_type() != torch::kUInt64) {
        throw std::invalid_argument("eval_tables must be torch.uint64");
    }
    if (eval_tables.dim() != 2) {
        throw std::invalid_argument("eval_tables must have shape (rows, N)");
    }
    if (!hp_spec::is_power_of_two_i64(eval_tables.size(1))) {
        throw std::invalid_argument("N must be a power of two");
    }
    if (eval_tables.size(1) < 2) {
        throw std::invalid_argument("N must be at least 2 for folding");
    }

    const c10::cuda::CUDAGuard device_guard(eval_tables.device());

    auto normal_tables = eval_tables.contiguous();
    auto normal_chal = challenge.to(eval_tables.options().dtype(torch::kUInt64)).contiguous();

    if (normal_chal.numel() < 1) {
        throw std::invalid_argument("challenge must contain at least one value");
    }

    int rows = static_cast<int>(normal_tables.size(0));
    int len = static_cast<int>(normal_tables.size(1));
    int half = len >> 1;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto tables_mont = torch::empty_like(normal_tables);
    auto chal_mont = torch::empty({1}, normal_tables.options());

    int conv_threads = SUMCHECK_THREADS;

    int conv_blocks_tables = std::min(
        SUMCHECK_MAX_BLOCKS,
        static_cast<int>(
            std::max<int64_t>(
                1,
                (normal_tables.numel() + conv_threads - 1) / conv_threads
            )
        )
    );

    convert_to_mont_u64_kernel<<<conv_blocks_tables, conv_threads, 0, stream>>>(
        reinterpret_cast<const u64*>(normal_tables.data_ptr<uint64_t>()),
        static_cast<size_t>(normal_tables.numel()),
        reinterpret_cast<u64*>(tables_mont.data_ptr<uint64_t>()));
    hp_spec::check_last_cuda("hashed u64 fold convert tables");

    convert_to_mont_u64_kernel<<<1, 1, 0, stream>>>(
        reinterpret_cast<const u64*>(normal_chal.data_ptr<uint64_t>()),
        static_cast<size_t>(1),
        reinterpret_cast<u64*>(chal_mont.data_ptr<uint64_t>()));
    hp_spec::check_last_cuda("hashed u64 fold convert challenge");

    auto next_mont = torch::empty({rows, half}, normal_tables.options());

    int upd_blocks = std::min(
        SUMCHECK_MAX_BLOCKS,
        std::max(
            1,
            (rows * half + SUMCHECK_THREADS - 1) / SUMCHECK_THREADS
        )
    );

    update_sumcheck_u64_kernel<<<upd_blocks, SUMCHECK_THREADS, 0, stream>>>(
        reinterpret_cast<const u64*>(tables_mont.data_ptr<uint64_t>()),
        rows,
        len,
        reinterpret_cast<const u64*>(chal_mont.data_ptr<uint64_t>()),
        0,
        reinterpret_cast<u64*>(next_mont.data_ptr<uint64_t>()));
    hp_spec::check_last_cuda("hashed u64 fold update");

    auto next_normal = torch::empty_like(next_mont);

    int conv_blocks_out = std::min(
        SUMCHECK_MAX_BLOCKS,
        static_cast<int>(
            std::max<int64_t>(
                1,
                (next_mont.numel() + conv_threads - 1) / conv_threads
            )
        )
    );

    convert_from_mont_u64_kernel<<<conv_blocks_out, conv_threads, 0, stream>>>(
        reinterpret_cast<const u64*>(next_mont.data_ptr<uint64_t>()),
        static_cast<size_t>(next_mont.numel()),
        reinterpret_cast<u64*>(next_normal.data_ptr<uint64_t>()));
    hp_spec::check_last_cuda("hashed u64 fold convert output");

    return next_normal;
}



// ============================================================================
// u128 Montgomery arithmetic experiment.
// Field: q = 2^128 - 159
// q = 0xffffffffffffffff_ffffffffffffff61
//
// Representation:
//   value = hi * 2^64 + lo
//
// This block adds a multiplication smoke-test entrypoint:
//   montgomery_u128_mul_test_cuda(a_lo, a_hi, b_lo, b_hi)
// ============================================================================

namespace mont_u128_exp {

using u64 = uint64_t;

struct u128x {
    u64 lo;
    u64 hi;
};

constexpr u64 Q_LO = 0xffffffffffffff61ULL;
constexpr u64 Q_HI = 0xffffffffffffffffULL;

// nprime = -q^{-1} mod 2^64, where q0 = Q_LO.
constexpr u64 NPRIME = 0xb5efe63d2eb11b5fULL;

// R^2 mod q, with R = 2^128.
// Since q = 2^128 - 159, R mod q = 159 and R^2 mod q = 159^2 = 25281.
constexpr u64 R2_LO = 0x00000000000062c1ULL;
constexpr u64 R2_HI = 0x0000000000000000ULL;

__device__ __forceinline__ bool ge_q(u128x a) {
    return (a.hi > Q_HI) || (a.hi == Q_HI && a.lo >= Q_LO);
}

__device__ __forceinline__ u128x sub_q(u128x a) {
    u64 lo = a.lo - Q_LO;
    u64 borrow = (a.lo < Q_LO) ? 1ULL : 0ULL;
    u64 hi = a.hi - Q_HI - borrow;
    return {lo, hi};
}

__device__ __forceinline__ u128x add_small(u128x a, u64 c) {
    u64 old = a.lo;
    a.lo += c;
    if (a.lo < old) {
        a.hi += 1ULL;
    }
    return a;
}

__device__ __forceinline__ u128x normalize(u128x a, u64 high_extra) {
    // After 2-limb REDC, result is less than 2q.
    // If the 129th bit is set, subtracting q is equivalent to adding 159.
    if (high_extra) {
        a = add_small(a, 159ULL);
    }

    if (ge_q(a)) {
        a = sub_q(a);
    }
    if (ge_q(a)) {
        a = sub_q(a);
    }
    return a;
}

__device__ __forceinline__ void add_limb(u64 t[6], int idx, u64 v) {
    u64 old = t[idx];
    t[idx] += v;
    u64 carry = (t[idx] < old) ? 1ULL : 0ULL;

    while (carry && idx + 1 < 6) {
        ++idx;
        old = t[idx];
        t[idx] += 1ULL;
        carry = (t[idx] == 0ULL) ? 1ULL : 0ULL;
    }
}

__device__ __forceinline__ void addmul_limb(u64 t[6], int idx, u64 a, u64 b) {
    u64 lo = a * b;
    u64 hi = __umul64hi(a, b);

    add_limb(t, idx, lo);
    add_limb(t, idx + 1, hi);
}

__device__ __forceinline__ void shift_right_limb(u64 t[6]) {
    t[0] = t[1];
    t[1] = t[2];
    t[2] = t[3];
    t[3] = t[4];
    t[4] = t[5];
    t[5] = 0ULL;
}

__device__ __forceinline__ u128x redc_256(u64 t[6]) {
    // Two-limb CIOS Montgomery REDC.
    // Base b = 2^64, modulus q has limbs Q_LO, Q_HI.
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        u64 m = t[0] * NPRIME;

        addmul_limb(t, 0, m, Q_LO);
        addmul_limb(t, 1, m, Q_HI);

        // t[0] is zero modulo 2^64 now; divide by b.
        shift_right_limb(t);
    }

    u128x out{t[0], t[1]};
    return normalize(out, t[2]);
}

__device__ __forceinline__ u128x mont_mul(u128x a, u128x b) {
    u64 t[6];

    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        t[i] = 0ULL;
    }

    addmul_limb(t, 0, a.lo, b.lo);
    addmul_limb(t, 1, a.lo, b.hi);
    addmul_limb(t, 1, a.hi, b.lo);
    addmul_limb(t, 2, a.hi, b.hi);

    return redc_256(t);
}

__device__ __forceinline__ u128x to_mont(u128x a) {
    u128x r2{R2_LO, R2_HI};
    return mont_mul(a, r2);
}

__device__ __forceinline__ u128x from_mont(u128x a_mont) {
    u64 t[6];

    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        t[i] = 0ULL;
    }

    t[0] = a_mont.lo;
    t[1] = a_mont.hi;

    return redc_256(t);
}

__device__ __forceinline__ u128x mul_normal(u128x a, u128x b) {
    u128x am = to_mont(a);
    u128x bm = to_mont(b);
    u128x cm = mont_mul(am, bm);
    return from_mont(cm);
}

__global__ void mul_test_kernel(
    const u64* __restrict__ a_lo,
    const u64* __restrict__ a_hi,
    const u64* __restrict__ b_lo,
    const u64* __restrict__ b_hi,
    u64* __restrict__ out_lo,
    u64* __restrict__ out_hi,
    size_t n) {

    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {

        u128x a{a_lo[i], a_hi[i]};
        u128x b{b_lo[i], b_hi[i]};

        u128x c = mul_normal(a, b);

        out_lo[i] = c.lo;
        out_hi[i] = c.hi;
    }
}

} // namespace mont_u128_exp

std::tuple<torch::Tensor, torch::Tensor> montgomery_u128_mul_test_cuda(
    torch::Tensor a_lo,
    torch::Tensor a_hi,
    torch::Tensor b_lo,
    torch::Tensor b_hi) {

    using namespace mont_u128_exp;

    if (!a_lo.is_cuda() || !a_hi.is_cuda() || !b_lo.is_cuda() || !b_hi.is_cuda()) {
        throw std::invalid_argument("all input tensors must be CUDA tensors");
    }

    if (a_lo.scalar_type() != torch::kUInt64 ||
        a_hi.scalar_type() != torch::kUInt64 ||
        b_lo.scalar_type() != torch::kUInt64 ||
        b_hi.scalar_type() != torch::kUInt64) {
        throw std::invalid_argument("all input tensors must be torch.uint64");
    }

    if (a_lo.sizes() != a_hi.sizes() ||
        a_lo.sizes() != b_lo.sizes() ||
        a_lo.sizes() != b_hi.sizes()) {
        throw std::invalid_argument("all input tensors must have the same shape");
    }

    const c10::cuda::CUDAGuard device_guard(a_lo.device());

    auto alo = a_lo.contiguous();
    auto ahi = a_hi.contiguous();
    auto blo = b_lo.contiguous();
    auto bhi = b_hi.contiguous();

    auto out_lo = torch::empty_like(alo);
    auto out_hi = torch::empty_like(ahi);

    constexpr int THREADS = 128;
    constexpr int MAX_BLOCKS = 4096;

    size_t n = static_cast<size_t>(alo.numel());

    int blocks = std::min<int>(
        MAX_BLOCKS,
        std::max<int>(1, static_cast<int>((n + THREADS - 1) / THREADS))
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    mul_test_kernel<<<blocks, THREADS, 0, stream>>>(
        reinterpret_cast<const u64*>(alo.data_ptr<uint64_t>()),
        reinterpret_cast<const u64*>(ahi.data_ptr<uint64_t>()),
        reinterpret_cast<const u64*>(blo.data_ptr<uint64_t>()),
        reinterpret_cast<const u64*>(bhi.data_ptr<uint64_t>()),
        reinterpret_cast<u64*>(out_lo.data_ptr<uint64_t>()),
        reinterpret_cast<u64*>(out_hi.data_ptr<uint64_t>()),
        n);

    hp_spec::check_last_cuda("montgomery_u128_mul_test_cuda");

    return std::make_tuple(out_lo, out_hi);
}



// ============================================================================
// Generic u128 full Montgomery-domain SumCheck.
// Field: q = 2^128 - 159.
// Tensor layout uses separate lo/hi uint64 tensors.
// ============================================================================

namespace mont_u128_exp {

constexpr int SUMCHECK_THREADS = 128;
constexpr int SUMCHECK_MAX_BLOCKS = 4096;
constexpr u64 MONT_ONE_LO = 159ULL;
constexpr u64 MONT_ONE_HI = 0ULL;

__device__ __forceinline__ bool ge_u128(u128x a, u128x b) {
    return (a.hi > b.hi) || (a.hi == b.hi && a.lo >= b.lo);
}

__device__ __forceinline__ u128x sub_u128_raw(u128x a, u128x b) {
    u64 lo = a.lo - b.lo;
    u64 borrow = (a.lo < b.lo) ? 1ULL : 0ULL;
    u64 hi = a.hi - b.hi - borrow;
    return {lo, hi};
}

__device__ __forceinline__ u128x add_small_with_carry(u128x a, u64 c, u64& extra) {
    u64 old_lo = a.lo;
    a.lo += c;
    if (a.lo < old_lo) {
        u64 old_hi = a.hi;
        a.hi += 1ULL;
        if (a.hi < old_hi) {
            extra += 1ULL;
        }
    }
    return a;
}

__device__ __forceinline__ u128x fold_extra_and_normalize(u128x a, u64 extra) {
    // Each extra 2^128 limb is equivalent to +159 mod q.
    while (extra) {
        u64 next_extra = 0;
        a = add_small_with_carry(a, 159ULL, next_extra);
        extra = next_extra;
    }

    if (ge_q(a)) {
        a = sub_q(a);
    }
    if (ge_q(a)) {
        a = sub_q(a);
    }
    return a;
}

__device__ __forceinline__ u128x add_mod_u128(u128x a, u128x b) {
    u128x out;
    out.lo = a.lo + b.lo;
    u64 carry_lo = (out.lo < a.lo) ? 1ULL : 0ULL;

    u64 hi1 = a.hi + b.hi;
    u64 carry_hi1 = (hi1 < a.hi) ? 1ULL : 0ULL;

    out.hi = hi1 + carry_lo;
    u64 carry_hi2 = (out.hi < hi1) ? 1ULL : 0ULL;

    return fold_extra_and_normalize(out, carry_hi1 + carry_hi2);
}

__device__ __forceinline__ u128x sub_small_u128(u128x a, u64 c) {
    u64 old_lo = a.lo;
    a.lo -= c;
    if (old_lo < c) {
        a.hi -= 1ULL;
    }
    return a;
}

__device__ __forceinline__ u128x sub_mod_u128(u128x a, u128x b) {
    if (ge_u128(a, b)) {
        return sub_u128_raw(a, b);
    }

    // Raw wrap gives a - b + 2^128. Since q = 2^128 - 159,
    // a - b + q = raw - 159.
    u128x raw = sub_u128_raw(a, b);
    return sub_small_u128(raw, 159ULL);
}

__device__ __forceinline__ u128x line_eval_mont_u128(u128x z, u128x o, u128x t) {
    u128x delta = sub_mod_u128(o, z);
    return add_mod_u128(z, mont_mul(t, delta));
}

__device__ __forceinline__ u128x load_line_mont_u128(
    const u64* __restrict__ tables_lo,
    const u64* __restrict__ tables_hi,
    int len,
    int row,
    int idx,
    int half,
    u128x x_mont) {

    size_t base = static_cast<size_t>(row) * static_cast<size_t>(len);

    u128x z{tables_lo[base + idx], tables_hi[base + idx]};
    u128x o{tables_lo[base + idx + half], tables_hi[base + idx + half]};

    return line_eval_mont_u128(z, o, x_mont);
}

__device__ __forceinline__ u128x eval_terms_at_x_mont_u128(
    const u64* __restrict__ tables_lo,
    const u64* __restrict__ tables_hi,
    int len,
    int idx,
    int half,
    const int32_t* __restrict__ term_offsets,
    const int32_t* __restrict__ term_vars,
    int num_terms,
    u128x x_mont) {

    u128x acc{0ULL, 0ULL};

    for (int term = 0; term < num_terms; ++term) {
        int start = term_offsets[term];
        int end = term_offsets[term + 1];

        u128x prod{MONT_ONE_LO, MONT_ONE_HI};

        for (int j = start; j < end; ++j) {
            int row = term_vars[j];

            u128x value = load_line_mont_u128(
                tables_lo,
                tables_hi,
                len,
                row,
                idx,
                half,
                x_mont
            );

            prod = mont_mul(prod, value);
        }

        acc = add_mod_u128(acc, prod);
    }

    return acc;
}

__device__ __forceinline__ u128x warp_reduce_add_mod_u128(u128x value) {
    unsigned mask = 0xffffffffu;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        u128x other;
        other.lo = __shfl_down_sync(mask, value.lo, offset);
        other.hi = __shfl_down_sync(mask, value.hi, offset);
        value = add_mod_u128(value, other);
    }

    return value;
}

__global__ void convert_to_mont_u128_kernel(
    const u64* __restrict__ in_lo,
    const u64* __restrict__ in_hi,
    size_t n,
    u64* __restrict__ out_lo,
    u64* __restrict__ out_hi) {

    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {

        u128x normal{in_lo[i], in_hi[i]};
        u128x mont = to_mont(normal);

        out_lo[i] = mont.lo;
        out_hi[i] = mont.hi;
    }
}

__global__ void convert_from_mont_u128_kernel(
    const u64* __restrict__ in_lo,
    const u64* __restrict__ in_hi,
    size_t n,
    u64* __restrict__ out_lo,
    u64* __restrict__ out_hi) {

    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {

        u128x mont{in_lo[i], in_hi[i]};
        u128x normal = from_mont(mont);

        out_lo[i] = normal.lo;
        out_hi[i] = normal.hi;
    }
}

__global__ void eval_terms_sumcheck_u128_kernel(
    const u64* __restrict__ tables_lo,
    const u64* __restrict__ tables_hi,
    int len,
    const int32_t* __restrict__ term_offsets,
    const int32_t* __restrict__ term_vars,
    int num_terms,
    int degree,
    u64* __restrict__ partials_lo,
    u64* __restrict__ partials_hi) {

    extern __shared__ u64 shared[];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = (blockDim.x + 31) >> 5;
    const int half = len >> 1;

    u128x local[8];

    #pragma unroll
    for (int x = 0; x < 8; ++x) {
        local[x] = {0ULL, 0ULL};
    }

    for (int i = blockIdx.x * blockDim.x + tid;
         i < half;
         i += blockDim.x * gridDim.x) {

        for (int x = 0; x <= degree; ++x) {
            u128x x_normal{static_cast<u64>(x), 0ULL};
            u128x x_mont = to_mont(x_normal);

            u128x y = eval_terms_at_x_mont_u128(
                tables_lo,
                tables_hi,
                len,
                i,
                half,
                term_offsets,
                term_vars,
                num_terms,
                x_mont
            );

            local[x] = add_mod_u128(local[x], y);
        }
    }

    for (int x = 0; x <= degree; ++x) {
        u128x reduced = warp_reduce_add_mod_u128(local[x]);

        if (lane == 0) {
            int slot = (x * num_warps + warp_id) * 2;
            shared[slot + 0] = reduced.lo;
            shared[slot + 1] = reduced.hi;
        }
    }

    __syncthreads();

    if (warp_id == 0) {
        for (int x = 0; x <= degree; ++x) {
            u128x value{0ULL, 0ULL};

            if (lane < num_warps) {
                int slot = (x * num_warps + lane) * 2;
                value.lo = shared[slot + 0];
                value.hi = shared[slot + 1];
            }

            value = warp_reduce_add_mod_u128(value);

            if (lane == 0) {
                size_t out_idx =
                    static_cast<size_t>(blockIdx.x) *
                    static_cast<size_t>(degree + 1) +
                    x;

                partials_lo[out_idx] = value.lo;
                partials_hi[out_idx] = value.hi;
            }
        }
    }
}

__global__ void reduce_sumcheck_u128_kernel(
    const u64* __restrict__ partials_lo,
    const u64* __restrict__ partials_hi,
    int num_blocks,
    int degree,
    u64* __restrict__ out_lo,
    u64* __restrict__ out_hi) {

    __shared__ u64 scratch_lo[SUMCHECK_THREADS];
    __shared__ u64 scratch_hi[SUMCHECK_THREADS];

    int x = blockIdx.x;
    int tid = threadIdx.x;

    u128x acc{0ULL, 0ULL};

    for (int b = tid; b < num_blocks; b += blockDim.x) {
        size_t idx =
            static_cast<size_t>(b) *
            static_cast<size_t>(degree + 1) +
            x;

        u128x v{partials_lo[idx], partials_hi[idx]};
        acc = add_mod_u128(acc, v);
    }

    scratch_lo[tid] = acc.lo;
    scratch_hi[tid] = acc.hi;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            u128x a{scratch_lo[tid], scratch_hi[tid]};
            u128x b{scratch_lo[tid + stride], scratch_hi[tid + stride]};
            u128x c = add_mod_u128(a, b);

            scratch_lo[tid] = c.lo;
            scratch_hi[tid] = c.hi;
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_lo[x] = scratch_lo[0];
        out_hi[x] = scratch_hi[0];
    }
}

__global__ void update_sumcheck_u128_kernel(
    const u64* __restrict__ in_lo,
    const u64* __restrict__ in_hi,
    int rows,
    int len,
    const u64* __restrict__ chals_lo,
    const u64* __restrict__ chals_hi,
    int round,
    u64* __restrict__ out_lo,
    u64* __restrict__ out_hi) {

    int half = len >> 1;
    size_t total = static_cast<size_t>(rows) * static_cast<size_t>(half);

    u128x r{chals_lo[round], chals_hi[round]};

    for (size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<size_t>(gridDim.x) * blockDim.x) {

        int row = static_cast<int>(linear / half);
        int i = static_cast<int>(
            linear - static_cast<size_t>(row) * static_cast<size_t>(half)
        );

        size_t in_base =
            static_cast<size_t>(row) * static_cast<size_t>(len);
        size_t out_base =
            static_cast<size_t>(row) * static_cast<size_t>(half);

        u128x z{in_lo[in_base + i], in_hi[in_base + i]};
        u128x o{in_lo[in_base + i + half], in_hi[in_base + i + half]};

        u128x y = line_eval_mont_u128(z, o, r);

        out_lo[out_base + i] = y.lo;
        out_hi[out_base + i] = y.hi;
    }
}

__global__ void claim0_from_output_u128_kernel(
    const u64* __restrict__ output_lo,
    const u64* __restrict__ output_hi,
    int degree,
    u64* __restrict__ claim_lo,
    u64* __restrict__ claim_hi) {

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        u128x c;

        if (degree >= 1) {
            u128x a{output_lo[0], output_hi[0]};
            u128x b{output_lo[1], output_hi[1]};
            c = add_mod_u128(a, b);
        } else {
            c = {output_lo[0], output_hi[0]};
        }

        claim_lo[0] = c.lo;
        claim_hi[0] = c.hi;
    }
}

} // namespace mont_u128_exp

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
sumcheck_terms_full_mont_u128_cuda(
    torch::Tensor eval_lo,
    torch::Tensor eval_hi,
    torch::Tensor challenges_lo,
    torch::Tensor challenges_hi,
    torch::Tensor term_offsets,
    torch::Tensor term_vars,
    uint64_t modulus_hi,
    uint64_t modulus_lo) {

    using namespace mont_u128_exp;

    if (modulus_hi != Q_HI || modulus_lo != Q_LO) {
        throw std::invalid_argument("u128 SumCheck currently requires q = 2^128 - 159");
    }

    if (!eval_lo.is_cuda() || !eval_hi.is_cuda()) {
        throw std::invalid_argument("eval_lo/eval_hi must be CUDA tensors");
    }
    if (eval_lo.scalar_type() != torch::kUInt64 ||
        eval_hi.scalar_type() != torch::kUInt64) {
        throw std::invalid_argument("eval_lo/eval_hi must be torch.uint64");
    }
    if (eval_lo.sizes() != eval_hi.sizes()) {
        throw std::invalid_argument("eval_lo/eval_hi must have the same shape");
    }
    if (eval_lo.dim() != 2) {
        throw std::invalid_argument("eval tensors must have shape (rows, N)");
    }
    if (!hp_spec::is_power_of_two_i64(eval_lo.size(1))) {
        throw std::invalid_argument("N must be a power of two");
    }

    const c10::cuda::CUDAGuard device_guard(eval_lo.device());

    auto offsets_cpu = term_offsets.to(torch::kCPU).to(torch::kInt32).contiguous();
    auto vars_cpu = term_vars.to(torch::kCPU).to(torch::kInt32).contiguous();

    if (offsets_cpu.dim() != 1 || offsets_cpu.size(0) < 2) {
        throw std::invalid_argument("term_offsets must be a 1D tensor with at least 2 entries");
    }
    if (vars_cpu.dim() != 1) {
        throw std::invalid_argument("term_vars must be a 1D tensor");
    }

    int num_terms = static_cast<int>(offsets_cpu.size(0)) - 1;
    const int32_t* offsets_ptr = offsets_cpu.data_ptr<int32_t>();
    const int32_t* vars_ptr = vars_cpu.data_ptr<int32_t>();

    int degree = 0;
    int max_row = -1;

    for (int t = 0; t < num_terms; ++t) {
        int start = offsets_ptr[t];
        int end = offsets_ptr[t + 1];

        if (start < 0 || end < start || end > vars_cpu.size(0)) {
            throw std::invalid_argument("invalid term_offsets");
        }

        int term_degree = end - start;
        degree = std::max(degree, term_degree);

        for (int j = start; j < end; ++j) {
            int row = vars_ptr[j];
            if (row < 0) {
                throw std::invalid_argument("term_vars contains a negative row index");
            }
            max_row = std::max(max_row, row);
        }
    }

    if (degree > 7) {
        throw std::invalid_argument("u128 generic path currently supports degree <= 7");
    }
    if (max_row >= eval_lo.size(0)) {
        throw std::invalid_argument("term_vars references a row outside eval tensors");
    }

    int rows = static_cast<int>(eval_lo.size(0));
    int initial_len = static_cast<int>(eval_lo.size(1));
    int rounds = hp_spec::log2_exact_i64(initial_len);

    auto normal_lo = eval_lo.contiguous();
    auto normal_hi = eval_hi.contiguous();

    auto ch_lo =
        challenges_lo.to(eval_lo.options().dtype(torch::kUInt64)).contiguous();
    auto ch_hi =
        challenges_hi.to(eval_lo.options().dtype(torch::kUInt64)).contiguous();

    if (ch_lo.dim() != 1 || ch_hi.dim() != 1 ||
        ch_lo.size(0) < rounds || ch_hi.size(0) < rounds) {
        throw std::invalid_argument("challenge tensors must have shape at least (log2(N),)");
    }

    auto int_opts_dev =
        torch::TensorOptions().device(eval_lo.device()).dtype(torch::kInt32);

    auto offsets_dev = term_offsets.to(int_opts_dev).contiguous();
    auto vars_dev = term_vars.to(int_opts_dev).contiguous();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto current_lo = torch::empty_like(normal_lo);
    auto current_hi = torch::empty_like(normal_hi);
    auto ch_mont_lo = torch::empty_like(ch_lo);
    auto ch_mont_hi = torch::empty_like(ch_hi);

    int conv_threads = SUMCHECK_THREADS;

    int conv_blocks_tables = std::min(
        SUMCHECK_MAX_BLOCKS,
        static_cast<int>(
            std::max<int64_t>(
                1,
                (normal_lo.numel() + conv_threads - 1) / conv_threads
            )
        )
    );

    int conv_blocks_chals = std::min(
        SUMCHECK_MAX_BLOCKS,
        static_cast<int>(
            std::max<int64_t>(
                1,
                (ch_lo.numel() + conv_threads - 1) / conv_threads
            )
        )
    );

    convert_to_mont_u128_kernel<<<conv_blocks_tables, conv_threads, 0, stream>>>(
        reinterpret_cast<const u64*>(normal_lo.data_ptr<uint64_t>()),
        reinterpret_cast<const u64*>(normal_hi.data_ptr<uint64_t>()),
        static_cast<size_t>(normal_lo.numel()),
        reinterpret_cast<u64*>(current_lo.data_ptr<uint64_t>()),
        reinterpret_cast<u64*>(current_hi.data_ptr<uint64_t>()));
    hp_spec::check_last_cuda("u128 terms convert tables");

    convert_to_mont_u128_kernel<<<conv_blocks_chals, conv_threads, 0, stream>>>(
        reinterpret_cast<const u64*>(ch_lo.data_ptr<uint64_t>()),
        reinterpret_cast<const u64*>(ch_hi.data_ptr<uint64_t>()),
        static_cast<size_t>(ch_lo.numel()),
        reinterpret_cast<u64*>(ch_mont_lo.data_ptr<uint64_t>()),
        reinterpret_cast<u64*>(ch_mont_hi.data_ptr<uint64_t>()));
    hp_spec::check_last_cuda("u128 terms convert challenges");

    auto output_mont_lo =
        torch::empty({rounds, degree + 1}, normal_lo.options());
    auto output_mont_hi =
        torch::empty({rounds, degree + 1}, normal_hi.options());

    for (int round = 0; round < rounds; ++round) {
        int len = static_cast<int>(current_lo.size(1));
        int half = len >> 1;

        int blocks = std::min(
            SUMCHECK_MAX_BLOCKS,
            std::max(1, (half + SUMCHECK_THREADS - 1) / SUMCHECK_THREADS)
        );

        auto partials_lo =
            torch::empty({blocks, degree + 1}, normal_lo.options());
        auto partials_hi =
            torch::empty({blocks, degree + 1}, normal_hi.options());

        size_t shmem =
            static_cast<size_t>(degree + 1) *
            SUMCHECK_THREADS *
            2 *
            sizeof(u64);

        eval_terms_sumcheck_u128_kernel<<<blocks, SUMCHECK_THREADS, shmem, stream>>>(
            reinterpret_cast<const u64*>(current_lo.data_ptr<uint64_t>()),
            reinterpret_cast<const u64*>(current_hi.data_ptr<uint64_t>()),
            len,
            reinterpret_cast<const int32_t*>(offsets_dev.data_ptr<int32_t>()),
            reinterpret_cast<const int32_t*>(vars_dev.data_ptr<int32_t>()),
            num_terms,
            degree,
            reinterpret_cast<u64*>(partials_lo.data_ptr<uint64_t>()),
            reinterpret_cast<u64*>(partials_hi.data_ptr<uint64_t>()));
        hp_spec::check_last_cuda("u128 terms eval");

        reduce_sumcheck_u128_kernel<<<degree + 1, SUMCHECK_THREADS, 0, stream>>>(
            reinterpret_cast<const u64*>(partials_lo.data_ptr<uint64_t>()),
            reinterpret_cast<const u64*>(partials_hi.data_ptr<uint64_t>()),
            blocks,
            degree,
            reinterpret_cast<u64*>(output_mont_lo[round].data_ptr<uint64_t>()),
            reinterpret_cast<u64*>(output_mont_hi[round].data_ptr<uint64_t>()));
        hp_spec::check_last_cuda("u128 terms reduce");

        if (round + 1 < rounds) {
            auto next_lo = torch::empty({rows, half}, normal_lo.options());
            auto next_hi = torch::empty({rows, half}, normal_hi.options());

            int upd_blocks = std::min(
                SUMCHECK_MAX_BLOCKS,
                std::max(
                    1,
                    (rows * half + SUMCHECK_THREADS - 1) / SUMCHECK_THREADS
                )
            );

            update_sumcheck_u128_kernel<<<upd_blocks, SUMCHECK_THREADS, 0, stream>>>(
                reinterpret_cast<const u64*>(current_lo.data_ptr<uint64_t>()),
                reinterpret_cast<const u64*>(current_hi.data_ptr<uint64_t>()),
                rows,
                len,
                reinterpret_cast<const u64*>(ch_mont_lo.data_ptr<uint64_t>()),
                reinterpret_cast<const u64*>(ch_mont_hi.data_ptr<uint64_t>()),
                round,
                reinterpret_cast<u64*>(next_lo.data_ptr<uint64_t>()),
                reinterpret_cast<u64*>(next_hi.data_ptr<uint64_t>()));
            hp_spec::check_last_cuda("u128 terms update");

            current_lo = next_lo;
            current_hi = next_hi;
        }
    }

    auto output_lo = torch::empty_like(output_mont_lo);
    auto output_hi = torch::empty_like(output_mont_hi);

    int conv_blocks_out = std::min(
        SUMCHECK_MAX_BLOCKS,
        static_cast<int>(
            std::max<int64_t>(
                1,
                (output_mont_lo.numel() + conv_threads - 1) / conv_threads
            )
        )
    );

    convert_from_mont_u128_kernel<<<conv_blocks_out, conv_threads, 0, stream>>>(
        reinterpret_cast<const u64*>(output_mont_lo.data_ptr<uint64_t>()),
        reinterpret_cast<const u64*>(output_mont_hi.data_ptr<uint64_t>()),
        static_cast<size_t>(output_mont_lo.numel()),
        reinterpret_cast<u64*>(output_lo.data_ptr<uint64_t>()),
        reinterpret_cast<u64*>(output_hi.data_ptr<uint64_t>()));
    hp_spec::check_last_cuda("u128 terms convert output");

    auto claim_lo = torch::empty({}, normal_lo.options());
    auto claim_hi = torch::empty({}, normal_hi.options());

    claim0_from_output_u128_kernel<<<1, 1, 0, stream>>>(
        reinterpret_cast<const u64*>(output_lo.data_ptr<uint64_t>()),
        reinterpret_cast<const u64*>(output_hi.data_ptr<uint64_t>()),
        degree,
        reinterpret_cast<u64*>(claim_lo.data_ptr<uint64_t>()),
        reinterpret_cast<u64*>(claim_hi.data_ptr<uint64_t>()));
    hp_spec::check_last_cuda("u128 terms claim0");

    return std::make_tuple(claim_lo, claim_hi, output_lo, output_hi);
}



// ============================================================================
// Specialized u128 full Montgomery-domain SumCheck.
// Field: q = 2^128 - 159.
// Uses fixed polynomial templates selected by poly_id.
// ============================================================================

namespace mont_u128_exp {

__device__ __forceinline__ u128x prod2_spec_u128(u128x a, u128x b) {
    return mont_mul(a, b);
}

__device__ __forceinline__ u128x prod3_spec_u128(u128x a, u128x b, u128x c) {
    return mont_mul(mont_mul(a, b), c);
}

__device__ __forceinline__ u128x prod4_spec_u128(u128x a, u128x b, u128x c, u128x d) {
    return mont_mul(mont_mul(mont_mul(a, b), c), d);
}

__device__ __forceinline__ u128x prod5_spec_u128(u128x a, u128x b, u128x c, u128x d, u128x e) {
    return mont_mul(mont_mul(mont_mul(mont_mul(a, b), c), d), e);
}

__device__ __forceinline__ u128x prod7_spec_u128(
    u128x a,
    u128x b,
    u128x c,
    u128x d,
    u128x e,
    u128x f,
    u128x g) {
    return mont_mul(mont_mul(prod5_spec_u128(a, b, c, d, e), f), g);
}

__device__ __forceinline__ u128x eval_hyperplonk_poly_at_x_mont_u128(
    const u64* __restrict__ tables_lo,
    const u64* __restrict__ tables_hi,
    int len,
    int idx,
    int half,
    int poly_id,
    u128x x_mont) {

    u128x acc{0ULL, 0ULL};

    if (poly_id == kPolyBaselineLinear) {
        // baseline_linear: a
        return load_line_mont_u128(tables_lo, tables_hi, len, 0, idx, half, x_mont);
    }

    if (poly_id == kPolyBaselineMul) {
        // baseline_mul: a*b
        u128x a = load_line_mont_u128(tables_lo, tables_hi, len, 0, idx, half, x_mont);
        u128x b = load_line_mont_u128(tables_lo, tables_hi, len, 1, idx, half, x_mont);
        return prod2_spec_u128(a, b);
    }

    if (poly_id == kPolyBaselineMulAdd) {
        // baseline_mul_add: a*b + c
        u128x a = load_line_mont_u128(tables_lo, tables_hi, len, 0, idx, half, x_mont);
        u128x b = load_line_mont_u128(tables_lo, tables_hi, len, 1, idx, half, x_mont);
        u128x c = load_line_mont_u128(tables_lo, tables_hi, len, 2, idx, half, x_mont);
        return add_mod_u128(prod2_spec_u128(a, b), c);
    }

    if (poly_id == kPolyBaselineCubicProduct) {
        // baseline_cubic_product: a*b*c
        u128x a = load_line_mont_u128(tables_lo, tables_hi, len, 0, idx, half, x_mont);
        u128x b = load_line_mont_u128(tables_lo, tables_hi, len, 1, idx, half, x_mont);
        u128x c = load_line_mont_u128(tables_lo, tables_hi, len, 2, idx, half, x_mont);
        return prod3_spec_u128(a, b, c);
    }

    if (poly_id == kPolyAdvancedA2B2C) {
        // advanced_a2b2c: a*a*b*b*c
        u128x a = load_line_mont_u128(tables_lo, tables_hi, len, 0, idx, half, x_mont);
        u128x b = load_line_mont_u128(tables_lo, tables_hi, len, 1, idx, half, x_mont);
        u128x c = load_line_mont_u128(tables_lo, tables_hi, len, 2, idx, half, x_mont);
        return prod5_spec_u128(a, a, b, b, c);
    }

    if (poly_id == kPolyAdvancedAbcPlusDe) {
        // advanced_abc_plus_de: a*b*c + d*e
        u128x a = load_line_mont_u128(tables_lo, tables_hi, len, 0, idx, half, x_mont);
        u128x b = load_line_mont_u128(tables_lo, tables_hi, len, 1, idx, half, x_mont);
        u128x c = load_line_mont_u128(tables_lo, tables_hi, len, 2, idx, half, x_mont);
        u128x d = load_line_mont_u128(tables_lo, tables_hi, len, 3, idx, half, x_mont);
        u128x e = load_line_mont_u128(tables_lo, tables_hi, len, 4, idx, half, x_mont);
        return add_mod_u128(prod3_spec_u128(a, b, c), prod2_spec_u128(d, e));
    }

    if (poly_id == kPolyAdvancedAbcgPlusDeg) {
        // advanced_abcg_plus_deg: a*b*c*g + d*e*g
        u128x a = load_line_mont_u128(tables_lo, tables_hi, len, 0, idx, half, x_mont);
        u128x b = load_line_mont_u128(tables_lo, tables_hi, len, 1, idx, half, x_mont);
        u128x c = load_line_mont_u128(tables_lo, tables_hi, len, 2, idx, half, x_mont);
        u128x d = load_line_mont_u128(tables_lo, tables_hi, len, 3, idx, half, x_mont);
        u128x e = load_line_mont_u128(tables_lo, tables_hi, len, 4, idx, half, x_mont);
        u128x g = load_line_mont_u128(tables_lo, tables_hi, len, 5, idx, half, x_mont);
        return add_mod_u128(prod4_spec_u128(a, b, c, g), prod3_spec_u128(d, e, g));
    }

    if (poly_id == kPolyVanillaGate) {
        // vanilla_gate:
        // qL*w1 + qR*w2 + qM*w1*w2 + neg_qO*w3 + qC
        u128x qL  = load_line_mont_u128(tables_lo, tables_hi, len, 0, idx, half, x_mont);
        u128x w1  = load_line_mont_u128(tables_lo, tables_hi, len, 1, idx, half, x_mont);
        u128x qR  = load_line_mont_u128(tables_lo, tables_hi, len, 2, idx, half, x_mont);
        u128x w2  = load_line_mont_u128(tables_lo, tables_hi, len, 3, idx, half, x_mont);
        u128x qM  = load_line_mont_u128(tables_lo, tables_hi, len, 4, idx, half, x_mont);
        u128x nqO = load_line_mont_u128(tables_lo, tables_hi, len, 5, idx, half, x_mont);
        u128x w3  = load_line_mont_u128(tables_lo, tables_hi, len, 6, idx, half, x_mont);
        u128x qC  = load_line_mont_u128(tables_lo, tables_hi, len, 7, idx, half, x_mont);

        acc = add_mod_u128(acc, prod2_spec_u128(qL, w1));
        acc = add_mod_u128(acc, prod2_spec_u128(qR, w2));
        acc = add_mod_u128(acc, prod3_spec_u128(qM, w1, w2));
        acc = add_mod_u128(acc, prod2_spec_u128(nqO, w3));
        acc = add_mod_u128(acc, qC);
        return acc;
    }

    if (poly_id == kPolyVanillaZero) {
        // vanilla_zero = vanilla_gate * fr, termwise.
        u128x qL  = load_line_mont_u128(tables_lo, tables_hi, len, 0, idx, half, x_mont);
        u128x w1  = load_line_mont_u128(tables_lo, tables_hi, len, 1, idx, half, x_mont);
        u128x qR  = load_line_mont_u128(tables_lo, tables_hi, len, 2, idx, half, x_mont);
        u128x w2  = load_line_mont_u128(tables_lo, tables_hi, len, 3, idx, half, x_mont);
        u128x qM  = load_line_mont_u128(tables_lo, tables_hi, len, 4, idx, half, x_mont);
        u128x nqO = load_line_mont_u128(tables_lo, tables_hi, len, 5, idx, half, x_mont);
        u128x w3  = load_line_mont_u128(tables_lo, tables_hi, len, 6, idx, half, x_mont);
        u128x qC  = load_line_mont_u128(tables_lo, tables_hi, len, 7, idx, half, x_mont);
        u128x fr  = load_line_mont_u128(tables_lo, tables_hi, len, 8, idx, half, x_mont);

        acc = add_mod_u128(acc, prod3_spec_u128(qL, w1, fr));
        acc = add_mod_u128(acc, prod3_spec_u128(qR, w2, fr));
        acc = add_mod_u128(acc, prod4_spec_u128(qM, w1, w2, fr));
        acc = add_mod_u128(acc, prod3_spec_u128(nqO, w3, fr));
        acc = add_mod_u128(acc, prod2_spec_u128(qC, fr));
        return acc;
    }

    if (poly_id == kPolyVanillaPerm) {
        // vanilla_perm:
        // (pi - p1*p2 + alpha_phi*D1*D2*D3 - alpha*N1*N2*N3) * fr
        // Signs/scalars are folded into input rows.
        u128x pi   = load_line_mont_u128(tables_lo, tables_hi, len, 0, idx, half, x_mont);
        u128x np1  = load_line_mont_u128(tables_lo, tables_hi, len, 1, idx, half, x_mont);
        u128x p2   = load_line_mont_u128(tables_lo, tables_hi, len, 2, idx, half, x_mont);
        u128x aphi = load_line_mont_u128(tables_lo, tables_hi, len, 3, idx, half, x_mont);
        u128x D1   = load_line_mont_u128(tables_lo, tables_hi, len, 4, idx, half, x_mont);
        u128x D2   = load_line_mont_u128(tables_lo, tables_hi, len, 5, idx, half, x_mont);
        u128x D3   = load_line_mont_u128(tables_lo, tables_hi, len, 6, idx, half, x_mont);
        u128x nN1  = load_line_mont_u128(tables_lo, tables_hi, len, 7, idx, half, x_mont);
        u128x N2   = load_line_mont_u128(tables_lo, tables_hi, len, 8, idx, half, x_mont);
        u128x N3   = load_line_mont_u128(tables_lo, tables_hi, len, 9, idx, half, x_mont);
        u128x fr   = load_line_mont_u128(tables_lo, tables_hi, len, 10, idx, half, x_mont);

        acc = add_mod_u128(acc, prod2_spec_u128(pi, fr));
        acc = add_mod_u128(acc, prod3_spec_u128(np1, p2, fr));
        acc = add_mod_u128(acc, prod5_spec_u128(aphi, D1, D2, D3, fr));
        acc = add_mod_u128(acc, prod4_spec_u128(nN1, N2, N3, fr));
        return acc;
    }

    if (poly_id == kPolyOpencheck6) {
        // opencheck_6: y1*k1 + ... + y6*k6
        #pragma unroll
        for (int r = 0; r < 6; ++r) {
            acc = add_mod_u128(
                acc,
                load_line_mont_u128(tables_lo, tables_hi, len, r, idx, half, x_mont)
            );
        }
        return acc;
    }

    if (poly_id == kPolyDegreeSweepDeg3 ||
        poly_id == kPolyDegreeSweepDeg5 ||
        poly_id == kPolyDegreeSweepDeg7) {
        // degree_sweep:
        // q1*w1 + q2*w2 + qH*w1^k*w2 + qC
        u128x q1 = load_line_mont_u128(tables_lo, tables_hi, len, 0, idx, half, x_mont);
        u128x w1 = load_line_mont_u128(tables_lo, tables_hi, len, 1, idx, half, x_mont);
        u128x q2 = load_line_mont_u128(tables_lo, tables_hi, len, 2, idx, half, x_mont);
        u128x w2 = load_line_mont_u128(tables_lo, tables_hi, len, 3, idx, half, x_mont);
        u128x qH = load_line_mont_u128(tables_lo, tables_hi, len, 4, idx, half, x_mont);
        u128x qC = load_line_mont_u128(tables_lo, tables_hi, len, 5, idx, half, x_mont);

        acc = add_mod_u128(acc, prod2_spec_u128(q1, w1));
        acc = add_mod_u128(acc, prod2_spec_u128(q2, w2));

        if (poly_id == kPolyDegreeSweepDeg3) {
            acc = add_mod_u128(acc, prod3_spec_u128(qH, w1, w2));
        } else if (poly_id == kPolyDegreeSweepDeg5) {
            acc = add_mod_u128(acc, prod5_spec_u128(qH, w1, w1, w1, w2));
        } else {
            acc = add_mod_u128(acc, prod7_spec_u128(qH, w1, w1, w1, w1, w1, w2));
        }

        acc = add_mod_u128(acc, qC);
        return acc;
    }

    return {0ULL, 0ULL};
}

__global__ void eval_hyperplonk_sumcheck_u128_kernel(
    const u64* __restrict__ tables_lo,
    const u64* __restrict__ tables_hi,
    int len,
    int poly_id,
    int degree,
    u64* __restrict__ partials_lo,
    u64* __restrict__ partials_hi) {

    extern __shared__ u64 shared[];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = (blockDim.x + 31) >> 5;
    const int half = len >> 1;

    u128x local[8];

    #pragma unroll
    for (int x = 0; x < 8; ++x) {
        local[x] = {0ULL, 0ULL};
    }

    for (int i = blockIdx.x * blockDim.x + tid;
         i < half;
         i += blockDim.x * gridDim.x) {

        for (int x = 0; x <= degree; ++x) {
            u128x x_normal{static_cast<u64>(x), 0ULL};
            u128x x_mont = to_mont(x_normal);

            u128x y = eval_hyperplonk_poly_at_x_mont_u128(
                tables_lo,
                tables_hi,
                len,
                i,
                half,
                poly_id,
                x_mont
            );

            local[x] = add_mod_u128(local[x], y);
        }
    }

    for (int x = 0; x <= degree; ++x) {
        u128x reduced = warp_reduce_add_mod_u128(local[x]);

        if (lane == 0) {
            int slot = (x * num_warps + warp_id) * 2;
            shared[slot + 0] = reduced.lo;
            shared[slot + 1] = reduced.hi;
        }
    }

    __syncthreads();

    if (warp_id == 0) {
        for (int x = 0; x <= degree; ++x) {
            u128x value{0ULL, 0ULL};

            if (lane < num_warps) {
                int slot = (x * num_warps + lane) * 2;
                value.lo = shared[slot + 0];
                value.hi = shared[slot + 1];
            }

            value = warp_reduce_add_mod_u128(value);

            if (lane == 0) {
                size_t out_idx =
                    static_cast<size_t>(blockIdx.x) *
                    static_cast<size_t>(degree + 1) +
                    x;

                partials_lo[out_idx] = value.lo;
                partials_hi[out_idx] = value.hi;
            }
        }
    }
}

} // namespace mont_u128_exp

std::tuple<torch::Tensor, torch::Tensor> sumcheck_hyperplonk_full_mont_u128_cuda(
    torch::Tensor eval_lo,
    torch::Tensor eval_hi,
    torch::Tensor challenges_lo,
    torch::Tensor challenges_hi,
    uint64_t modulus_hi,
    uint64_t modulus_lo,
    int64_t poly_id_64) {

    using namespace mont_u128_exp;

    if (modulus_hi != Q_HI || modulus_lo != Q_LO) {
        throw std::invalid_argument("u128 specialized SumCheck currently requires q = 2^128 - 159");
    }

    int poly_id = static_cast<int>(poly_id_64);


    int degree = degree_for_poly(poly_id);
    int rows = rows_for_poly(poly_id);

    if (!eval_lo.is_cuda() || !eval_hi.is_cuda()) {
        throw std::invalid_argument("eval_lo/eval_hi must be CUDA tensors");
    }
    if (eval_lo.scalar_type() != torch::kUInt64 ||
        eval_hi.scalar_type() != torch::kUInt64) {
        throw std::invalid_argument("eval_lo/eval_hi must be torch.uint64");
    }
    if (eval_lo.sizes() != eval_hi.sizes()) {
        throw std::invalid_argument("eval_lo/eval_hi must have the same shape");
    }
    if (eval_lo.dim() != 2) {
        throw std::invalid_argument("eval tensors must have shape (rows, N)");
    }
    if (eval_lo.size(0) < rows) {
        throw std::invalid_argument("eval tensors have too few rows for requested poly_id");
    }
    if (!hp_spec::is_power_of_two_i64(eval_lo.size(1))) {
        throw std::invalid_argument("N must be a power of two");
    }

    const c10::cuda::CUDAGuard device_guard(eval_lo.device());

    auto normal_lo = eval_lo.narrow(0, 0, rows).contiguous();
    auto normal_hi = eval_hi.narrow(0, 0, rows).contiguous();

    auto ch_lo =
        challenges_lo.to(eval_lo.options().dtype(torch::kUInt64)).contiguous();
    auto ch_hi =
        challenges_hi.to(eval_lo.options().dtype(torch::kUInt64)).contiguous();

    int initial_len = static_cast<int>(normal_lo.size(1));
    int rounds = hp_spec::log2_exact_i64(initial_len);

    if (ch_lo.dim() != 1 || ch_hi.dim() != 1 ||
        ch_lo.size(0) < rounds || ch_hi.size(0) < rounds) {
        throw std::invalid_argument("challenge tensors must have shape at least (log2(N),)");
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto current_lo = torch::empty_like(normal_lo);
    auto current_hi = torch::empty_like(normal_hi);
    auto ch_mont_lo = torch::empty_like(ch_lo);
    auto ch_mont_hi = torch::empty_like(ch_hi);

    int conv_threads = SUMCHECK_THREADS;

    int conv_blocks_tables = std::min(
        SUMCHECK_MAX_BLOCKS,
        static_cast<int>(
            std::max<int64_t>(
                1,
                (normal_lo.numel() + conv_threads - 1) / conv_threads
            )
        )
    );

    int conv_blocks_chals = std::min(
        SUMCHECK_MAX_BLOCKS,
        static_cast<int>(
            std::max<int64_t>(
                1,
                (ch_lo.numel() + conv_threads - 1) / conv_threads
            )
        )
    );

    convert_to_mont_u128_kernel<<<conv_blocks_tables, conv_threads, 0, stream>>>(
        reinterpret_cast<const u64*>(normal_lo.data_ptr<uint64_t>()),
        reinterpret_cast<const u64*>(normal_hi.data_ptr<uint64_t>()),
        static_cast<size_t>(normal_lo.numel()),
        reinterpret_cast<u64*>(current_lo.data_ptr<uint64_t>()),
        reinterpret_cast<u64*>(current_hi.data_ptr<uint64_t>()));
    hp_spec::check_last_cuda("u128 spec convert tables");

    convert_to_mont_u128_kernel<<<conv_blocks_chals, conv_threads, 0, stream>>>(
        reinterpret_cast<const u64*>(ch_lo.data_ptr<uint64_t>()),
        reinterpret_cast<const u64*>(ch_hi.data_ptr<uint64_t>()),
        static_cast<size_t>(ch_lo.numel()),
        reinterpret_cast<u64*>(ch_mont_lo.data_ptr<uint64_t>()),
        reinterpret_cast<u64*>(ch_mont_hi.data_ptr<uint64_t>()));
    hp_spec::check_last_cuda("u128 spec convert challenges");

    auto output_mont_lo =
        torch::empty({rounds, degree + 1}, normal_lo.options());
    auto output_mont_hi =
        torch::empty({rounds, degree + 1}, normal_hi.options());

    for (int round = 0; round < rounds; ++round) {
        int len = static_cast<int>(current_lo.size(1));
        int half = len >> 1;

        int blocks = std::min(
            SUMCHECK_MAX_BLOCKS,
            std::max(1, (half + SUMCHECK_THREADS - 1) / SUMCHECK_THREADS)
        );

        auto partials_lo =
            torch::empty({blocks, degree + 1}, normal_lo.options());
        auto partials_hi =
            torch::empty({blocks, degree + 1}, normal_hi.options());

        size_t shmem =
            static_cast<size_t>(degree + 1) *
            SUMCHECK_THREADS *
            2 *
            sizeof(u64);

        eval_hyperplonk_sumcheck_u128_kernel<<<blocks, SUMCHECK_THREADS, shmem, stream>>>(
            reinterpret_cast<const u64*>(current_lo.data_ptr<uint64_t>()),
            reinterpret_cast<const u64*>(current_hi.data_ptr<uint64_t>()),
            len,
            poly_id,
            degree,
            reinterpret_cast<u64*>(partials_lo.data_ptr<uint64_t>()),
            reinterpret_cast<u64*>(partials_hi.data_ptr<uint64_t>()));
        hp_spec::check_last_cuda("u128 spec eval");

        reduce_sumcheck_u128_kernel<<<degree + 1, SUMCHECK_THREADS, 0, stream>>>(
            reinterpret_cast<const u64*>(partials_lo.data_ptr<uint64_t>()),
            reinterpret_cast<const u64*>(partials_hi.data_ptr<uint64_t>()),
            blocks,
            degree,
            reinterpret_cast<u64*>(output_mont_lo[round].data_ptr<uint64_t>()),
            reinterpret_cast<u64*>(output_mont_hi[round].data_ptr<uint64_t>()));
        hp_spec::check_last_cuda("u128 spec reduce");

        if (round + 1 < rounds) {
            auto next_lo = torch::empty({rows, half}, normal_lo.options());
            auto next_hi = torch::empty({rows, half}, normal_hi.options());

            int upd_blocks = std::min(
                SUMCHECK_MAX_BLOCKS,
                std::max(
                    1,
                    (rows * half + SUMCHECK_THREADS - 1) / SUMCHECK_THREADS
                )
            );

            update_sumcheck_u128_kernel<<<upd_blocks, SUMCHECK_THREADS, 0, stream>>>(
                reinterpret_cast<const u64*>(current_lo.data_ptr<uint64_t>()),
                reinterpret_cast<const u64*>(current_hi.data_ptr<uint64_t>()),
                rows,
                len,
                reinterpret_cast<const u64*>(ch_mont_lo.data_ptr<uint64_t>()),
                reinterpret_cast<const u64*>(ch_mont_hi.data_ptr<uint64_t>()),
                round,
                reinterpret_cast<u64*>(next_lo.data_ptr<uint64_t>()),
                reinterpret_cast<u64*>(next_hi.data_ptr<uint64_t>()));
            hp_spec::check_last_cuda("u128 spec update");

            current_lo = next_lo;
            current_hi = next_hi;
        }
    }

    auto output_lo = torch::empty_like(output_mont_lo);
    auto output_hi = torch::empty_like(output_mont_hi);

    int conv_blocks_out = std::min(
        SUMCHECK_MAX_BLOCKS,
        static_cast<int>(
            std::max<int64_t>(
                1,
                (output_mont_lo.numel() + conv_threads - 1) / conv_threads
            )
        )
    );

    convert_from_mont_u128_kernel<<<conv_blocks_out, conv_threads, 0, stream>>>(
        reinterpret_cast<const u64*>(output_mont_lo.data_ptr<uint64_t>()),
        reinterpret_cast<const u64*>(output_mont_hi.data_ptr<uint64_t>()),
        static_cast<size_t>(output_mont_lo.numel()),
        reinterpret_cast<u64*>(output_lo.data_ptr<uint64_t>()),
        reinterpret_cast<u64*>(output_hi.data_ptr<uint64_t>()));
    hp_spec::check_last_cuda("u128 spec convert output");

    return std::make_tuple(output_lo, output_hi);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sumcheck_hyperplonk_full_mont_u128_cuda",
          &sumcheck_hyperplonk_full_mont_u128_cuda,
          "Specialized full Montgomery-domain u128 SumCheck");

    m.def("sumcheck_terms_full_mont_u128_cuda",
          &sumcheck_terms_full_mont_u128_cuda,
          "Generic full Montgomery-domain u128 SumCheck");

    m.def("montgomery_u128_mul_test_cuda",
          &montgomery_u128_mul_test_cuda,
          "u128 Montgomery modular multiplication smoke test");

    m.def("sumcheck_terms_full_mont_u64_round_eval_cuda",
          &sumcheck_terms_full_mont_u64_round_eval_cuda,
          "One-round generic full Montgomery-domain u64 SumCheck evaluation");
    m.def("fold_full_mont_u64_cuda",
          &fold_full_mont_u64_cuda,
          "One-round full Montgomery-domain u64 MLE fold");

    m.def("sumcheck_hyperplonk_full_mont_u64_cuda",
          &sumcheck_hyperplonk_full_mont_u64_cuda,
          "Specialized full Montgomery-domain u64 SumCheck");

    m.def("sumcheck_terms_full_mont_u64_cuda",
          &sumcheck_terms_full_mont_u64_cuda,
          "Generic full Montgomery-domain u64 SumCheck");

    m.def("montgomery_u64_mul_test_cuda",
          &montgomery_u64_mul_test_cuda,
          "u64 Montgomery modular multiplication smoke test");

    m.def("sumcheck_terms_full_mont_u32_cuda",
          &sumcheck_terms_full_mont_u32_cuda,
          "Experimental full Montgomery-domain generic u32 SumCheck");

    m.def("sumcheck_hyperplonk_full_mont_u32_cuda",
          &sumcheck_hyperplonk_full_mont_u32_cuda,
          "Experimental full Montgomery-domain specialized u32 SumCheck");


}
