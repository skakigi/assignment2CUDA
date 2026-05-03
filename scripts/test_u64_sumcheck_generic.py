import sys
import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import sumcheck_cuda_ext
import paper_poly_compare as base

Q = (1 << 64) - (1 << 32) + 1


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

    return np.array(out, dtype=np.uint64)


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
    tables_np = rng.integers(0, 1000, size=(rows, n), dtype=np.uint64)
    chals_np = rng.integers(0, 1000, size=(nv,), dtype=np.uint64)

    ref = ref_sumcheck_terms(tables_np, chals_np, terms)

    offsets, flat = flatten_terms(terms)

    tables = torch.from_numpy(tables_np).cuda()
    chals = torch.from_numpy(chals_np).cuda()
    term_offsets = torch.tensor(offsets, dtype=torch.int32)
    term_vars = torch.tensor(flat, dtype=torch.int32)

    _claim, out = sumcheck_cuda_ext.sumcheck_terms_full_mont_u64_cuda(
        tables,
        chals,
        term_offsets,
        term_vars,
        0,
        Q,
    )

    got = out.cpu().numpy()

    if not np.array_equal(got, ref):
        diff = np.argwhere(got != ref)[0]
        idx = tuple(int(x) for x in diff)
        raise AssertionError(
            f"{poly} nv={nv} mismatch at {idx}: got={int(got[idx])} expected={int(ref[idx])}"
        )

    print(f"OK {poly} nv={nv} shape={got.shape}")


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
    ]

    for nv in [4, 8]:
        for i, poly in enumerate(polys):
            run_one(poly, nv, seed=1000 + 17 * nv + i)

    print("u64 generic full-Montgomery SumCheck smoke OK")


if __name__ == "__main__":
    main()
