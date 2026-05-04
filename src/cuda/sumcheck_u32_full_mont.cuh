#pragma once

// Maintained u32 full-Montgomery SumCheck kernels and host wrappers.
// Included once by sumcheck_native.cu.

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
