#pragma once

// Maintained u64 full-Montgomery SumCheck kernels, hashed helpers, and host wrappers.
// Included once by sumcheck_native.cu.

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
