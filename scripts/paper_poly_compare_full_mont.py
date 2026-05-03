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


def build_full_mont_generic_call(tables, chals, terms):
    offsets, flat = base.flatten_terms(terms)
    term_offsets = torch.tensor(offsets, dtype=torch.int32)
    term_vars = torch.tensor(flat, dtype=torch.int32)

    def fn():
        _claim0, out = sumcheck_cuda_ext.sumcheck_terms_full_mont_u32_cuda(
            tables,
            chals,
            term_offsets,
            term_vars,
            Q32,
        )
        return out

    return fn


def build_full_mont_specialized_call(tables, chals, poly_name):
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

    required = [
        "sumcheck_terms_full_mont_u32_cuda",
        "sumcheck_hyperplonk_full_mont_u32_cuda",
    ]
    for name in required:
        if not hasattr(sumcheck_cuda_ext, name):
            raise RuntimeError(f"extension does not expose {name}")

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
        "regular_generic_ms",
        "full_generic_ms",
        "regular_spec_ms",
        "full_spec_ms",
        "full_generic_speed",
        "full_spec_speed",
        "shape",
    ]

    widths = [24, 10, 4, 18, 16, 16, 13, 18, 15, 12]

    for nv in num_vars_list:
        print()
        print(f"num_vars = {nv}")
        print()
        print(fmt_header(headers, widths))
        print("-+-".join("-" * w for w in widths))

        for poly in poly_names:
            if poly not in base.POLYS:
                raise ValueError(f"unknown polynomial template {poly}")

            terms = base.POLYS[poly]
            tables, chals = base.make_inputs(
                terms,
                nv,
                args.seed + nv + base.POLY_IDS[poly] * 100,
            )

            regular_generic_fn = base.build_generic_call(tables, chals, terms)
            full_generic_fn = build_full_mont_generic_call(tables, chals, terms)
            regular_spec_fn = base.build_specialized_call(tables, chals, poly)
            full_spec_fn = build_full_mont_specialized_call(tables, chals, poly)

            _rgf, rgmed, _rgp90, rgout = bench(regular_generic_fn, args.warmup, args.runs)
            _fgf, fgmed, _fgp90, fgout = bench(full_generic_fn, args.warmup, args.runs)
            _rsf, rsmed, _rsp90, rsout = bench(regular_spec_fn, args.warmup, args.runs)
            _fsf, fsmed, _fsp90, fsout = bench(full_spec_fn, args.warmup, args.runs)

            if args.check:
                ref = rgout.detach().cpu()
                fg = fgout.detach().cpu()
                ws = rsout.detach().cpu()
                fs = fsout.detach().cpu()

                for label, val in [
                    ("full_generic", fg),
                    ("regular_spec", ws),
                    ("full_spec", fs),
                ]:
                    if not torch.equal(ref, val):
                        diff = (ref != val).nonzero()[0].tolist()
                        raise AssertionError(
                            f"regular_generic/{label} mismatch {poly} nv={nv} first={diff}"
                        )

            n = 1 << nv

            row = [
                poly,
                f"{n:d}",
                f"{base.degree_for_terms(terms):d}",
                f"{rgmed:.3f}",
                f"{fgmed:.3f}",
                f"{rsmed:.3f}",
                f"{fsmed:.3f}",
                f"{rgmed / fgmed:.2f}x",
                f"{rgmed / fsmed:.2f}x",
                str(tuple(rgout.shape)),
            ]

            print(fmt_row(row, widths))

        print()


if __name__ == "__main__":
    main()
