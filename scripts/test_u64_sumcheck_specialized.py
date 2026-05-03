import sys
import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import sumcheck_cuda_ext
import paper_poly_compare as base

Q = (1 << 64) - (1 << 32) + 1


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

    tables = torch.from_numpy(tables_np).cuda()
    chals = torch.from_numpy(chals_np).cuda()

    offsets, flat = flatten_terms(terms)
    term_offsets = torch.tensor(offsets, dtype=torch.int32)
    term_vars = torch.tensor(flat, dtype=torch.int32)

    _claim, generic_out = sumcheck_cuda_ext.sumcheck_terms_full_mont_u64_cuda(
        tables,
        chals,
        term_offsets,
        term_vars,
        0,
        Q,
    )

    spec_out = sumcheck_cuda_ext.sumcheck_hyperplonk_full_mont_u64_cuda(
        tables,
        chals,
        0,
        Q,
        base.POLY_IDS[poly],
    )

    g = generic_out.cpu().numpy()
    s = spec_out.cpu().numpy()

    if not np.array_equal(g, s):
        diff = np.argwhere(g != s)[0]
        idx = tuple(int(x) for x in diff)
        raise AssertionError(
            f"{poly} nv={nv} mismatch at {idx}: "
            f"generic={int(g[idx])} specialized={int(s[idx])}"
        )

    print(f"OK {poly} nv={nv} shape={g.shape}")


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
            run_one(poly, nv, seed=2000 + 17 * nv + i)

    print("u64 specialized full-Montgomery SumCheck smoke OK")


if __name__ == "__main__":
    main()
