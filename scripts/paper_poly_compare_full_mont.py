import argparse
import os
import statistics
import sys

import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import sumcheck_cuda_ext
import paper_poly_compare as base


Q32 = 4294967291


def time_call(fn):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), out


def bench(fn, warmup, runs):
    first_ms, out = time_call(fn)

    for _ in range(warmup):
        time_call(fn)

    times = []
    for _ in range(runs):
        ms, out = time_call(fn)
        times.append(ms)

    med = statistics.median(times)
    p90 = sorted(times)[int(0.9 * (len(times) - 1))]
    return first_ms, med, p90, out


def build_full_mont_call(tables, chals, poly_name):
    poly_id = base.POLY_IDS[poly_name]

    def fn():
        return sumcheck_cuda_ext.sumcheck_hyperplonk_full_mont_u32_cuda(
            tables,
            chals,
            Q32,
            poly_id,
        )

    return fn


def fmt_row(values, widths):
    return " | ".join(str(v).rjust(w) for v, w in zip(values, widths))


def fmt_header(values, widths):
    return " | ".join(str(v).center(w) for v, w in zip(values, widths))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-vars", default="20")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--polys", default="all")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not hasattr(sumcheck_cuda_ext, "sumcheck_hyperplonk_full_mont_u32_cuda"):
        raise RuntimeError("extension does not expose sumcheck_hyperplonk_full_mont_u32_cuda")

    num_vars_list = [int(x) for x in args.num_vars.split(",")]

    if args.polys.strip().lower() == "all":
        poly_names = list(base.DEFAULT_ORDER)
    else:
        poly_names = [x.strip() for x in args.polys.split(",") if x.strip()]

    print("device:", torch.cuda.get_device_name())
    print("SC_EVAL_VARIANT:", os.environ.get("SC_EVAL_VARIANT", "unset/default"))
    print("backend funcs:", [x for x in dir(sumcheck_cuda_ext) if "sumcheck" in x])
    print()

    headers = [
        "template",
        "N",
        "deg",
        "generic_ms",
        "wrapped_spec_ms",
        "full_mont_ms",
        "wrapped_speed",
        "full_mont_speed",
        "full_mont_p90",
        "shape",
    ]

    widths = [24, 10, 4, 12, 16, 13, 13, 15, 14, 12]

    for nv in num_vars_list:
        print()
        print(f"num_vars = {nv}")
        print()
        print(fmt_header(headers, widths))
        print("-+-".join("-" * w for w in widths))

        for poly in poly_names:
            terms = base.POLYS[poly]
            tables, chals = base.make_inputs(
                terms,
                nv,
                args.seed + nv + base.POLY_IDS[poly] * 100,
            )

            generic_fn = base.build_generic_call(tables, chals, terms)
            wrapped_spec_fn = base.build_specialized_call(tables, chals, poly)
            full_mont_fn = build_full_mont_call(tables, chals, poly)

            _gf, gmed, _gp90, gout = bench(generic_fn, args.warmup, args.runs)
            _sf, smed, _sp90, sout = bench(wrapped_spec_fn, args.warmup, args.runs)
            _mf, mmed, mp90, mout = bench(full_mont_fn, args.warmup, args.runs)

            if args.check:
                g = gout.detach().cpu()
                s = sout.detach().cpu()
                m = mout.detach().cpu()

                if not torch.equal(g, s):
                    diff = (g != s).nonzero()[0].tolist()
                    raise AssertionError(f"generic/wrapped mismatch {poly} nv={nv} first={diff}")

                if not torch.equal(g, m):
                    diff = (g != m).nonzero()[0].tolist()
                    raise AssertionError(f"generic/full_mont mismatch {poly} nv={nv} first={diff}")

            n = 1 << nv

            row = [
                poly,
                f"{n:d}",
                f"{base.degree_for_terms(terms):d}",
                f"{gmed:.3f}",
                f"{smed:.3f}",
                f"{mmed:.3f}",
                f"{gmed / smed:.2f}x",
                f"{gmed / mmed:.2f}x",
                f"{mp90:.3f}",
                str(tuple(gout.shape)),
            ]

            print(fmt_row(row, widths))

        print()


if __name__ == "__main__":
    main()
