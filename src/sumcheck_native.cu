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

















constexpr int kEvalTStride = 4;









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

#include "cuda/sumcheck_u32_full_mont.cuh"


#include "cuda/sumcheck_u64_full_mont.cuh"


#include "cuda/sumcheck_u128_full_mont.cuh"


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

    m.def("sumcheck_hyperplonk_full_mont_u64_round_eval_cuda",
          &sumcheck_hyperplonk_full_mont_u64_round_eval_cuda,
          "Specialized full Montgomery-domain u64 single-round eval for hashed transcript");

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
