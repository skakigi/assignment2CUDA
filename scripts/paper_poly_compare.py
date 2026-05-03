import argparse
import os
import statistics
import sys

import torch

sys.path.insert(0, "src")
import sumcheck_cuda_ext


Q32 = 4294967291


POLYS = {
    # Assignment baseline expressions.
    "baseline_linear": [[0]],                       # a
    "baseline_mul": [[0, 1]],                       # a*b
    "baseline_mul_add": [[0, 1], [2]],              # a*b + c
    "baseline_cubic_product": [[0, 1, 2]],          # a*b*c

    "advanced_a2b2c": [[0, 0, 1, 1, 2]],       # a*a*b*b*c
    "advanced_abc_plus_de": [[0, 1, 2], [3, 4]], # a*b*c + d*e
    "advanced_abcg_plus_deg": [[0, 1, 2, 5], [3, 4, 5]], # a*b*c*g + d*e*g

    # HyperPlonk / zkSpeed-style protocol templates.
    "vanilla_gate": [
        [0, 1],
        [2, 3],
        [4, 1, 3],
        [5, 6],
        [7],
    ],
    "vanilla_zero": [
        [0, 1, 8],
        [2, 3, 8],
        [4, 1, 3, 8],
        [5, 6, 8],
        [7, 8],
    ],
    "vanilla_perm": [
        [0, 10],
        [1, 2, 10],
        [3, 4, 5, 6, 10],
        [7, 8, 9, 10],
    ],
    "opencheck_6": [
        [0], [1], [2], [3], [4], [5],
    ],
    # Custom-gate degree sweep:
    #   q1*w1 + q2*w2 + qH*w1^k*w2 + qC
    "degree_sweep_deg3": [
        [0, 1],
        [2, 3],
        [4, 1, 3],
        [5],
    ],
    "degree_sweep_deg5": [
        [0, 1],
        [2, 3],
        [4, 1, 1, 1, 3],
        [5],
    ],
    "degree_sweep_deg7": [
        [0, 1],
        [2, 3],
        [4, 1, 1, 1, 1, 1, 3],
        [5],
    ],
}


FUNCTIONS = {
    "baseline_linear": "a",
    "baseline_mul": "a*b",
    "baseline_mul_add": "a*b + c",
    "baseline_cubic_product": "a*b*c",


    "advanced_a2b2c": "a*a*b*b*c",
    "advanced_abc_plus_de": "a*b*c + d*e",
    "advanced_abcg_plus_deg": "a*b*c*g + d*e*g",
    "vanilla_gate": "qL*w1 + qR*w2 + qM*w1*w2 - qO*w3 + qC",
    "vanilla_zero": "(qL*w1 + qR*w2 + qM*w1*w2 - qO*w3 + qC)*fr",
    "vanilla_perm": "(pi - p1*p2 + alpha_phi*D1*D2*D3 - alpha*N1*N2*N3)*fr",
    "opencheck_6": "y1*k1 + y2*k2 + ... + y6*k6",
    "degree_sweep_deg3": "q1*w1 + q2*w2 + qH*w1*w2 + qC",
    "degree_sweep_deg5": "q1*w1 + q2*w2 + qH*w1^3*w2 + qC",
    "degree_sweep_deg7": "q1*w1 + q2*w2 + qH*w1^5*w2 + qC",
}


POLY_IDS = {
    "vanilla_gate": 0,
    "vanilla_zero": 1,
    "vanilla_perm": 2,
    "opencheck_6": 3,
    "degree_sweep_deg3": 5,
    "degree_sweep_deg5": 6,
    "degree_sweep_deg7": 7,
    "baseline_linear": 8,
    "baseline_mul": 9,
    "baseline_mul_add": 10,
    "baseline_cubic_product": 11,

    "advanced_a2b2c": 12,
    "advanced_abc_plus_de": 13,
    "advanced_abcg_plus_deg": 14,
}


DEFAULT_ORDER = [
    "baseline_linear",
    "baseline_mul",
    "baseline_mul_add",
    "baseline_cubic_product",

    "advanced_a2b2c",
    "advanced_abc_plus_de",
    "advanced_abcg_plus_deg",
    "vanilla_gate",
    "vanilla_zero",
    "vanilla_perm",
    "opencheck_6",
    "degree_sweep_deg3",
    "degree_sweep_deg5",
    "degree_sweep_deg7",
]


def flatten_terms(terms):
    offsets = [0]
    flat = []
    for term in terms:
        flat.extend(term)
        offsets.append(len(flat))
    return offsets, flat


def rows_for_terms(terms):
    return max(max(t) for t in terms) + 1


def degree_for_terms(terms):
    return max(len(t) for t in terms)


def time_call(fn):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), out


def make_inputs(terms, num_vars, seed):
    n = 1 << num_vars
    rows = rows_for_terms(terms)

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    tables_cpu = torch.randint(
        0, 1000, (rows, n), dtype=torch.int64, generator=g
    ).to(torch.uint32)
    chals_cpu = torch.randint(
        0, 1000, (num_vars,), dtype=torch.int64, generator=g
    ).to(torch.uint32)

    return tables_cpu.cuda().contiguous(), chals_cpu.cuda().contiguous()


def build_generic_call(tables, chals, terms):
    offsets, flat = flatten_terms(terms)
    term_offsets = torch.tensor(offsets, dtype=torch.int32)
    term_vars = torch.tensor(flat, dtype=torch.int32)

    def fn():
        _claim0, out = sumcheck_cuda_ext.sumcheck_terms_full_mont_u32_cuda(
            tables, chals, term_offsets, term_vars, Q32
        )
        return out

    return fn


def build_specialized_call(tables, chals, poly_name):
    poly_id = POLY_IDS[poly_name]

    def fn():
        return sumcheck_cuda_ext.sumcheck_hyperplonk_full_mont_u32_cuda(
            tables, chals, Q32, poly_id
        )

    return fn


def bench(fn, warmup, runs):
    first_ms, out = time_call(fn)

    for _ in range(warmup):
        time_call(fn)

    times = []
    for _ in range(runs):
        ms, _ = time_call(fn)
        times.append(ms)

    med = statistics.median(times)
    p90 = sorted(times)[int(0.9 * (len(times) - 1))]
    return first_ms, med, p90, out


def _fmt_row(values, widths):
    return " | ".join(str(v).rjust(w) for v, w in zip(values, widths))


def _fmt_header(values, widths):
    return " | ".join(str(v).center(w) for v, w in zip(values, widths))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-vars", default="20")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--polys", default=",".join(DEFAULT_ORDER))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    num_vars_list = [int(x) for x in args.num_vars.split(",")]
    if args.polys.strip().lower() == "all":
        poly_names = list(DEFAULT_ORDER)
    else:
        poly_names = [x.strip() for x in args.polys.split(",") if x.strip()]

    required = [
        "sumcheck_terms_full_mont_u32_cuda",
        "sumcheck_hyperplonk_full_mont_u32_cuda",
    ]
    missing = [name for name in required if not hasattr(sumcheck_cuda_ext, name)]
    if missing:
        raise RuntimeError(f"Missing required full Montgomery entrypoints: {missing}")

    print("device:", torch.cuda.get_device_name())
    print("SC_EVAL_VARIANT:", os.environ.get("SC_EVAL_VARIANT", "unset/default"))
    print("backend funcs:", [x for x in dir(sumcheck_cuda_ext) if "sumcheck" in x])
    print("active backend: full Montgomery domain")
    print()

    headers = [
        "template",
        "function",
        "N",
        "rows",
        "terms",
        "deg",
        "generic_ms",
        "spec_ms",
        "speedup",
        "generic_p90",
        "spec_p90",
        "generic_Mpts/s",
        "spec_Mpts/s",
        "shape",
    ]

    widths = [
        24,
        62,
        10,
        5,
        5,
        4,
        12,
        10,
        8,
        12,
        10,
        15,
        13,
        12,
    ]

    for nv in num_vars_list:
        print()
        print(f"num_vars = {nv}")
        print()
        print(_fmt_header(headers, widths))
        print("-+-".join("-" * w for w in widths))

        for poly in poly_names:
            if poly not in POLYS:
                raise ValueError(f"unknown polynomial template {poly}; choices={sorted(POLYS)}")

            terms = POLYS[poly]
            tables, chals = make_inputs(
                terms,
                nv,
                args.seed + nv + POLY_IDS[poly] * 100,
            )

            generic_fn = build_generic_call(tables, chals, terms)
            spec_fn = build_specialized_call(tables, chals, poly)

            generic_first, generic_med, generic_p90, generic_out = bench(
                generic_fn,
                args.warmup,
                args.runs,
            )
            spec_first, spec_med, spec_p90, spec_out = bench(
                spec_fn,
                args.warmup,
                args.runs,
            )

            if args.check:
                g = generic_out.detach().cpu()
                sp = spec_out.detach().cpu()
                if not torch.equal(g, sp):
                    diff = (g != sp).nonzero()
                    first = diff[0].tolist()
                    raise AssertionError(
                        f"generic/specialized mismatch for {poly} nv={nv}; "
                        f"first mismatch {first}; generic={g[tuple(first)].item()} "
                        f"specialized={sp[tuple(first)].item()}"
                    )

            n = 1 << nv
            speedup = generic_med / spec_med if spec_med > 0 else float("inf")
            generic_mpts = (n / generic_med) / 1000.0
            spec_mpts = (n / spec_med) / 1000.0

            row = [
                poly,
                FUNCTIONS[poly],
                f"{n:d}",
                f"{rows_for_terms(terms):d}",
                f"{len(terms):d}",
                f"{degree_for_terms(terms):d}",
                f"{generic_med:.3f}",
                f"{spec_med:.3f}",
                f"{speedup:.2f}x",
                f"{generic_p90:.3f}",
                f"{spec_p90:.3f}",
                f"{generic_mpts:.2f}",
                f"{spec_mpts:.2f}",
                str(tuple(generic_out.shape)),
            ]

            print(_fmt_row(row, widths))

        print()


if __name__ == "__main__":
    main()
