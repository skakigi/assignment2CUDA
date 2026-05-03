import sys
import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import sumcheck_cuda_ext
import paper_poly_compare as base

Q = (1 << 128) - 159
MASK64 = (1 << 64) - 1
Q_HI = (Q >> 64) & MASK64
Q_LO = Q & MASK64


def split_u128_array(vals):
    lo = np.empty(len(vals), dtype=np.uint64)
    hi = np.empty(len(vals), dtype=np.uint64)
    for i, x in enumerate(vals):
        lo[i] = np.uint64(x & MASK64)
        hi[i] = np.uint64((x >> 64) & MASK64)
    return lo, hi


def split_matrix(mat):
    rows, cols = mat.shape
    lo = np.empty((rows, cols), dtype=np.uint64)
    hi = np.empty((rows, cols), dtype=np.uint64)
    for r in range(rows):
        for c in range(cols):
            x = int(mat[r, c])
            lo[r, c] = np.uint64(x & MASK64)
            hi[r, c] = np.uint64((x >> 64) & MASK64)
    return lo, hi


def combine_matrix(lo, hi):
    out = []
    for r in range(lo.shape[0]):
        row = []
        for c in range(lo.shape[1]):
            row.append((int(hi[r, c]) << 64) | int(lo[r, c]))
        out.append(row)
    return out


def line_eval(z, o, t):
    return (z + t * ((o - z) % Q)) % Q


def ref_sumcheck_terms(tables, challenges, terms):
    current = [[int(x) % Q for x in row] for row in tables]
    rounds = len(challenges)
    degree = max(len(t) for t in terms)
    out = []

    for r in range(rounds):
        n = len(current[0])
        half = n // 2
        row_out = []

        for x in range(degree + 1):
            total = 0
            for i in range(half):
                acc = 0
                for term in terms:
                    prod = 1
                    for row in term:
                        z = current[row][i]
                        o = current[row][i + half]
                        prod = (prod * line_eval(z, o, x)) % Q
                    acc = (acc + prod) % Q
                total = (total + acc) % Q
            row_out.append(total)

        out.append(row_out)

        chal = int(challenges[r]) % Q
        nxt = []
        for row in current:
            nxt_row = []
            for i in range(half):
                nxt_row.append(line_eval(row[i], row[i + half], chal))
            nxt.append(nxt_row)
        current = nxt

    return out


def flatten_terms(terms):
    offsets = [0]
    flat = []
    for term in terms:
        flat.extend(term)
        offsets.append(len(flat))
    return offsets, flat


def run_one(poly, nv, seed):
    terms = base.POLYS[poly]
    rows = base.rows_for_terms(terms)
    n = 1 << nv

    rng = np.random.default_rng(seed)

    # Keep values small for simple CPU reference/debugging.
    tables_obj = np.empty((rows, n), dtype=object)
    for r in range(rows):
        for i in range(n):
            tables_obj[r, i] = int(rng.integers(0, 1000))

    chals = [int(rng.integers(0, 1000)) for _ in range(nv)]

    ref = ref_sumcheck_terms(tables_obj, chals, terms)

    eval_lo_np, eval_hi_np = split_matrix(tables_obj)
    ch_lo_np, ch_hi_np = split_u128_array(chals)

    eval_lo = torch.from_numpy(eval_lo_np).cuda()
    eval_hi = torch.from_numpy(eval_hi_np).cuda()
    ch_lo = torch.from_numpy(ch_lo_np).cuda()
    ch_hi = torch.from_numpy(ch_hi_np).cuda()

    offsets, flat = flatten_terms(terms)
    term_offsets = torch.tensor(offsets, dtype=torch.int32)
    term_vars = torch.tensor(flat, dtype=torch.int32)

    _clo, _chi, out_lo, out_hi = sumcheck_cuda_ext.sumcheck_terms_full_mont_u128_cuda(
        eval_lo,
        eval_hi,
        ch_lo,
        ch_hi,
        term_offsets,
        term_vars,
        Q_HI,
        Q_LO,
    )

    got = combine_matrix(out_lo.cpu().numpy(), out_hi.cpu().numpy())

    if got != ref:
        for i in range(len(ref)):
            for j in range(len(ref[i])):
                if got[i][j] != ref[i][j]:
                    raise AssertionError(
                        f"{poly} nv={nv} mismatch at ({i},{j}): "
                        f"got={got[i][j]} expected={ref[i][j]}"
                    )

    print(f"OK {poly} nv={nv} shape=({len(got)}, {len(got[0])})")


def main():
    polys = [
        "baseline_linear",
        "baseline_mul",
        "baseline_mul_add",
        "baseline_cubic_product",
        "vanilla_gate",
        "vanilla_zero",
        "vanilla_perm",
        "opencheck_6",
        "degree_sweep_deg3",
        "degree_sweep_deg5",
        "degree_sweep_deg7",
        "advanced_a2b2c",
        "advanced_abc_plus_de",
        "advanced_abcg_plus_deg",
    ]

    for nv in [4, 6]:
        for i, poly in enumerate(polys):
            run_one(poly, nv, seed=3000 + 17 * nv + i)

    print("u128 generic full-Montgomery SumCheck smoke OK")


if __name__ == "__main__":
    main()
