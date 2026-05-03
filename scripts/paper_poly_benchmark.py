import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, "src")
import sumcheck_cuda_ext


Q32 = 4294967291


POLYS = {
    # Assignment-style
    "a": [[0]],
    "a*b": [[0, 1]],
    "a*b+c": [[0, 1], [2]],
    "a*b*c": [[0, 1, 2]],

    # Plonk / HyperPlonk-inspired
    # qL*w1 + qR*w2 + qM*w1*w2 - qO*w3 + qC
    # neg/scalar coefficients are assumed folded into generated MLE rows.
    "vanilla_gate": [
        [0, 1],        # qL*w1
        [2, 3],        # qR*w2
        [4, 1, 3],     # qM*w1*w2
        [5, 6],        # neg_qO*w3
        [7],           # qC
    ],

    # (qL*w1 + qR*w2 + qM*w1*w2 - qO*w3 + qC) * fr
    "vanilla_zero": [
        [0, 1, 8],
        [2, 3, 8],
        [4, 1, 3, 8],
        [5, 6, 8],
        [7, 8],
    ],

    # (pi - p1*p2 + alpha*phi*D1*D2*D3 - alpha*N1*N2*N3) * fr
    "vanilla_perm": [
        [0, 10],             # pi*fr
        [1, 2, 10],          # neg_p1*p2*fr
        [3, 4, 5, 6, 10],    # alpha_phi*D1*D2*D3*fr
        [7, 8, 9, 10],       # neg_alpha_N1*N2*N3*fr
    ],

    # y1*k1 + ... + y6*k6, coefficients folded into rows
    "opencheck_6": [
        [0], [1], [2], [3], [4], [5],
    ],

    # Jellyfish-ish high-degree ZeroCheck structural template.
    # This intentionally repeats rows to represent powers like w1^5.
    "jellyfish_zero": [
        [0, 1, 17],                    # q1*w1*fr
        [2, 3, 17],                    # q2*w2*fr
        [4, 5, 17],                    # q3*w3*fr
        [6, 7, 17],                    # q4*w4*fr
        [8, 1, 3, 17],                 # qM1*w1*w2*fr
        [9, 5, 7, 17],                 # qM2*w3*w4*fr
        [10, 1, 1, 1, 1, 1, 17],       # qH1*w1^5*fr
        [11, 3, 3, 3, 3, 3, 17],       # qH2*w2^5*fr
        [12, 5, 5, 5, 5, 5, 17],       # qH3*w3^5*fr
        [13, 7, 7, 7, 7, 7, 17],       # qH4*w4^5*fr
        [14, 15, 17],                  # neg_qO*w5*fr
        [16, 1, 3, 5, 7, 17],          # qECC*w1*w2*w3*w4*fr
        [18, 17],                      # qC*fr
    ],
}


def flatten_terms(terms):
    offsets = [0]
    flat = []
    for term in terms:
        flat.extend(term)
        offsets.append(len(flat))
    return offsets, flat


def num_rows_for_terms(terms):
    return max(max(term) for term in terms) + 1


def time_call(fn):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), out


def bench_poly(name, terms, num_vars, warmup, runs, seed):
    n = 1 << num_vars
    rows = num_rows_for_terms(terms)
    offsets, flat = flatten_terms(terms)

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    tables_cpu = torch.randint(
        low=0,
        high=1000,
        size=(rows, n),
        dtype=torch.int64,
        generator=g,
    ).to(torch.uint32)

    chals_cpu = torch.randint(
        low=0,
        high=1000,
        size=(num_vars,),
        dtype=torch.int64,
        generator=g,
    ).to(torch.uint32)

    tables = tables_cpu.cuda().contiguous()
    chals = chals_cpu.cuda().contiguous()
    term_offsets = torch.tensor(offsets, dtype=torch.int32)
    term_vars = torch.tensor(flat, dtype=torch.int32)

    def fn():
        claim0, out = sumcheck_cuda_ext.sumcheck_terms_u32_cuda(
            tables,
            chals,
            term_offsets,
            term_vars,
            Q32,
        )
        return out

    first_ms, out = time_call(fn)

    for _ in range(warmup):
        time_call(fn)

    times = []
    for _ in range(runs):
        ms, _ = time_call(fn)
        times.append(ms)

    med = statistics.median(times)
    p90 = sorted(times)[int(0.9 * (len(times) - 1))]
    degree = max(len(t) for t in terms)

    return {
        "name": name,
        "N": n,
        "rows": rows,
        "terms": len(terms),
        "degree": degree,
        "first_ms": first_ms,
        "median_ms": med,
        "p90_ms": p90,
        "mpts": (n / med) / 1000.0,
        "shape": tuple(out.shape),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-vars", type=str, default="20")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--polys", type=str, default=",".join(POLYS.keys()))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    num_vars_list = [int(x) for x in args.num_vars.split(",")]
    poly_names = [x.strip() for x in args.polys.split(",") if x.strip()]

    print("device:", torch.cuda.get_device_name())
    print("SC_EVAL_VARIANT:", os.environ.get("SC_EVAL_VARIANT", "unset/default"))
    print("backend funcs:", [x for x in dir(sumcheck_cuda_ext) if "sumcheck" in x])
    print()

    header = (
        f"{'num_vars':>8} {'poly':<18} {'N':>9} {'rows':>5} "
        f"{'terms':>5} {'deg':>4} {'first_ms':>9} {'median_ms':>10} "
        f"{'p90_ms':>8} {'Mpts/s':>9} {'out_shape':>12}"
    )
    print(header)
    print("-" * len(header))

    for nv in num_vars_list:
        for name in poly_names:
            if name not in POLYS:
                raise ValueError(f"unknown polynomial {name}; choices={sorted(POLYS)}")
            r = bench_poly(name, POLYS[name], nv, args.warmup, args.runs, args.seed + nv)

            print(
                f"{nv:8d} {name:<18} {r['N']:9d} {r['rows']:5d} "
                f"{r['terms']:5d} {r['degree']:4d} {r['first_ms']:9.3f} "
                f"{r['median_ms']:10.3f} {r['p90_ms']:8.3f} "
                f"{r['mpts']:9.2f} {str(r['shape']):>12}"
            )


if __name__ == "__main__":
    main()
