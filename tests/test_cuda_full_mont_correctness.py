from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import paper_poly_compare as poly_defs  # noqa: E402


Q32 = 4294967291
Q64 = (1 << 64) - (1 << 32) + 1
Q128 = (1 << 128) - 159
MASK64 = (1 << 64) - 1
Q128_LO = Q128 & MASK64
Q128_HI = (Q128 >> 64) & MASK64

SMALL_POLYS = [
    "baseline_linear",
    "baseline_mul",
    "baseline_mul_add",
    "baseline_cubic_product",
    "advanced_abc_plus_de",
    "vanilla_gate",
    "vanilla_zero",
    "vanilla_perm",
    "opencheck_6",
    "degree_sweep_deg7",
]

SMALL_NUM_VARS = [1, 2, 4, 6]


def flatten_terms(terms):
    offsets = [0]
    flat = []
    for term in terms:
        flat.extend(term)
        offsets.append(len(flat))
    return offsets, flat


def split_u128(values: np.ndarray):
    vals = np.asarray(values, dtype=object)
    lo = np.vectorize(lambda x: int(x) & MASK64, otypes=[np.uint64])(vals)
    hi = np.vectorize(lambda x: (int(x) >> 64) & MASK64, otypes=[np.uint64])(vals)
    return lo.astype(np.uint64), hi.astype(np.uint64)


def pack_u128(lo, hi):
    lo_np = lo.detach().cpu().numpy()
    hi_np = hi.detach().cpu().numpy()
    packed = []
    for i in range(lo_np.shape[0]):
        row = []
        for j in range(lo_np.shape[1]):
            row.append(int(lo_np[i, j]) + (int(hi_np[i, j]) << 64))
        packed.append(row)
    return packed


def split_half_sumcheck_terms_reference(tables, challenges, terms, modulus):
    if modulus <= 1:
        raise ValueError("modulus must be > 1")

    cur = [[int(x) % modulus for x in row] for row in tables]
    rs = [int(r) % modulus for r in challenges]

    if not cur:
        raise ValueError("expected at least one MLE table")
    n = len(cur[0])
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError("table length must be a power of two")
    if any(len(row) != n for row in cur):
        raise ValueError("all MLE tables must have equal length")
    if len(rs) != n.bit_length() - 1:
        raise ValueError("challenge count must equal log2(table length)")

    degree = max(len(term) for term in terms)
    transcript = []

    for r in rs:
        cur_len = len(cur[0])
        half = cur_len // 2
        round_evals = [0] * (degree + 1)

        for i in range(half):
            for x in range(degree + 1):
                acc_terms = 0
                for term in terms:
                    prod = 1
                    for row_idx in term:
                        a = cur[row_idx][i]
                        b = cur[row_idx][i + half]
                        val = (a + x * ((b - a) % modulus)) % modulus
                        prod = (prod * val) % modulus
                    acc_terms = (acc_terms + prod) % modulus
                round_evals[x] = (round_evals[x] + acc_terms) % modulus

        transcript.append(round_evals)

        nxt = [[0] * half for _ in cur]
        for row_idx, row in enumerate(cur):
            for i in range(half):
                a = row[i]
                b = row[i + half]
                nxt[row_idx][i] = (a + r * ((b - a) % modulus)) % modulus
        cur = nxt

    return transcript


def initial_claim_from_transcript(transcript, modulus):
    return (int(transcript[0][0]) + int(transcript[0][1])) % modulus


def make_inputs(bits, terms, num_vars, seed):
    import torch

    rng = np.random.default_rng(seed)
    rows = poly_defs.rows_for_terms(terms)
    n = 1 << num_vars

    tables_np = rng.integers(0, 1000, size=(rows, n), dtype=np.uint64)
    chals_np = rng.integers(0, 1000, size=(num_vars,), dtype=np.uint64)

    if bits == 32:
        tables = torch.from_numpy(tables_np.astype(np.uint32)).cuda().contiguous()
        chals = torch.from_numpy(chals_np.astype(np.uint32)).cuda().contiguous()
        return tables, chals, tables_np.tolist(), chals_np.tolist(), Q32

    if bits == 64:
        tables = torch.from_numpy(tables_np.astype(np.uint64)).cuda().contiguous()
        chals = torch.from_numpy(chals_np.astype(np.uint64)).cuda().contiguous()
        return tables, chals, tables_np.tolist(), chals_np.tolist(), Q64

    if bits == 128:
        tables_lo, tables_hi = split_u128(tables_np)
        chals_lo, chals_hi = split_u128(chals_np)
        tables = (
            torch.from_numpy(tables_lo).cuda().contiguous(),
            torch.from_numpy(tables_hi).cuda().contiguous(),
        )
        chals = (
            torch.from_numpy(chals_lo).cuda().contiguous(),
            torch.from_numpy(chals_hi).cuda().contiguous(),
        )
        return tables, chals, tables_np.tolist(), chals_np.tolist(), Q128

    raise ValueError(bits)


def call_generic(native, bits, tables, chals, terms):
    import torch

    offsets, flat = flatten_terms(terms)
    term_offsets = torch.tensor(offsets, dtype=torch.int32)
    term_vars = torch.tensor(flat, dtype=torch.int32)

    if bits == 32:
        claim0, out = native.sumcheck_terms_full_mont_u32_cuda(
            tables, chals, term_offsets, term_vars, Q32
        )
        return claim0, out

    if bits == 64:
        claim0, out = native.sumcheck_terms_full_mont_u64_cuda(
            tables, chals, term_offsets, term_vars, 0, Q64
        )
        return claim0, out

    if bits == 128:
        eval_lo, eval_hi = tables
        ch_lo, ch_hi = chals
        claim_lo, claim_hi, out_lo, out_hi = native.sumcheck_terms_full_mont_u128_cuda(
            eval_lo,
            eval_hi,
            ch_lo,
            ch_hi,
            term_offsets,
            term_vars,
            Q128_HI,
            Q128_LO,
        )
        return (claim_lo, claim_hi), (out_lo, out_hi)

    raise ValueError(bits)


def call_specialized(native, bits, tables, chals, poly_name):
    poly_id = poly_defs.POLY_IDS[poly_name]

    if bits == 32:
        return native.sumcheck_hyperplonk_full_mont_u32_cuda(
            tables, chals, Q32, poly_id
        )

    if bits == 64:
        return native.sumcheck_hyperplonk_full_mont_u64_cuda(
            tables, chals, 0, Q64, poly_id
        )

    if bits == 128:
        eval_lo, eval_hi = tables
        ch_lo, ch_hi = chals
        return native.sumcheck_hyperplonk_full_mont_u128_cuda(
            eval_lo,
            eval_hi,
            ch_lo,
            ch_hi,
            Q128_HI,
            Q128_LO,
            poly_id,
        )

    raise ValueError(bits)


def tensor_to_python_matrix(out, bits):
    if bits == 128:
        lo, hi = out
        return pack_u128(lo, hi)
    arr = out.detach().cpu().numpy()
    return [[int(x) for x in row] for row in arr]


def claim_to_int(claim, bits):
    if bits == 128:
        lo, hi = claim
        return int(lo.detach().cpu().item()) + (int(hi.detach().cpu().item()) << 64)
    return int(claim.detach().cpu().item())


def assert_matrix_equal_mod(got, expected, modulus):
    assert len(got) == len(expected)
    for r, (got_row, exp_row) in enumerate(zip(got, expected)):
        assert len(got_row) == len(exp_row)
        for c, (g, e) in enumerate(zip(got_row, exp_row)):
            assert int(g) % modulus == int(e) % modulus, (r, c, g, e)


@pytest.fixture(scope="module")
def cuda_native():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    try:
        import sumcheck_cuda_ext as native
    except Exception as exc:
        pytest.skip(f"native CUDA extension is not built/importable: {exc}")

    required = [
        "sumcheck_terms_full_mont_u32_cuda",
        "sumcheck_hyperplonk_full_mont_u32_cuda",
        "sumcheck_terms_full_mont_u64_cuda",
        "sumcheck_hyperplonk_full_mont_u64_cuda",
        "sumcheck_terms_full_mont_u128_cuda",
        "sumcheck_hyperplonk_full_mont_u128_cuda",
    ]
    missing = [name for name in required if not hasattr(native, name)]
    if missing:
        pytest.skip(f"native extension missing full-Montgomery symbols: {missing}")

    return native


@pytest.mark.parametrize("bits", [32, 64, 128])
@pytest.mark.parametrize("num_vars", SMALL_NUM_VARS)
@pytest.mark.parametrize("poly_name", SMALL_POLYS)
def test_generic_full_mont_cuda_matches_cpu_reference(cuda_native, bits, num_vars, poly_name):
    terms = poly_defs.POLYS[poly_name]
    seed = bits * 100_000 + num_vars * 1_000 + poly_defs.POLY_IDS[poly_name]

    tables, chals, tables_cpu, chals_cpu, modulus = make_inputs(
        bits, terms, num_vars, seed
    )

    expected = split_half_sumcheck_terms_reference(
        tables_cpu, chals_cpu, terms, modulus
    )
    expected_claim0 = initial_claim_from_transcript(expected, modulus)

    claim0, got_out = call_generic(cuda_native, bits, tables, chals, terms)
    got = tensor_to_python_matrix(got_out, bits)

    assert claim_to_int(claim0, bits) % modulus == expected_claim0
    assert_matrix_equal_mod(got, expected, modulus)


@pytest.mark.parametrize("bits", [32, 64, 128])
@pytest.mark.parametrize("num_vars", SMALL_NUM_VARS)
@pytest.mark.parametrize("poly_name", SMALL_POLYS)
def test_specialized_full_mont_cuda_matches_cpu_reference(cuda_native, bits, num_vars, poly_name):
    terms = poly_defs.POLYS[poly_name]
    seed = 17 + bits * 100_000 + num_vars * 1_000 + poly_defs.POLY_IDS[poly_name]

    tables, chals, tables_cpu, chals_cpu, modulus = make_inputs(
        bits, terms, num_vars, seed
    )

    expected = split_half_sumcheck_terms_reference(
        tables_cpu, chals_cpu, terms, modulus
    )
    got_out = call_specialized(cuda_native, bits, tables, chals, poly_name)
    got = tensor_to_python_matrix(got_out, bits)

    assert_matrix_equal_mod(got, expected, modulus)


@pytest.mark.parametrize("bits", [32, 64, 128])
@pytest.mark.parametrize("num_vars", [4, 6])
@pytest.mark.parametrize("poly_name", SMALL_POLYS)
def test_generic_and_specialized_full_mont_agree(cuda_native, bits, num_vars, poly_name):
    terms = poly_defs.POLYS[poly_name]
    seed = 31 + bits * 100_000 + num_vars * 1_000 + poly_defs.POLY_IDS[poly_name]

    tables, chals, _tables_cpu, _chals_cpu, modulus = make_inputs(
        bits, terms, num_vars, seed
    )

    _claim0, generic_out = call_generic(cuda_native, bits, tables, chals, terms)
    spec_out = call_specialized(cuda_native, bits, tables, chals, poly_name)

    generic = tensor_to_python_matrix(generic_out, bits)
    specialized = tensor_to_python_matrix(spec_out, bits)

    assert_matrix_equal_mod(generic, specialized, modulus)
