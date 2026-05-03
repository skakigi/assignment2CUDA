import argparse
import os
import statistics
import sys

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import sumcheck_cuda_ext
import paper_poly_compare as base


Q32 = 4294967291
Q64 = (1 << 64) - (1 << 32) + 1


def flatten_terms(terms):
    offsets = [0]
    flat = []
    for term in terms:
        flat.extend(term)
        offsets.append(len(flat))
    return offsets, flat


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


def make_inputs(bits, terms, num_vars, seed):
    n = 1 << num_vars
    rows = base.rows_for_terms(terms)

    if bits == 32:
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)

        tables_cpu = torch.randint(
            0, 1000, (rows, n), dtype=torch.int64, generator=g
        ).to(torch.uint32)
        chals_cpu = torch.randint(
            0, 1000, (num_vars,), dtype=torch.int64, generator=g
        ).to(torch.uint32)

        return tables_cpu.cuda().contiguous(), chals_cpu.cuda().contiguous()

    if bits == 64:
        rng = np.random.default_rng(seed)
        tables_np = rng.integers(0, 1000, size=(rows, n), dtype=np.uint64)
        chals_np = rng.integers(0, 1000, size=(num_vars,), dtype=np.uint64)

        tables = torch.from_numpy(tables_np).cuda().contiguous()
        chals = torch.from_numpy(chals_np).cuda().contiguous()

        return tables, chals

    raise ValueError(f"unsupported bits={bits}")


def build_generic_call(bits, tables, chals, terms):
    offsets, flat = flatten_terms(terms)
    term_offsets = torch.tensor(offsets, dtype=torch.int32)
    term_vars = torch.tensor(flat, dtype=torch.int32)

    if bits == 32:
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

    if bits == 64:
        def fn():
            _claim0, out = sumcheck_cuda_ext.sumcheck_terms_full_mont_u64_cuda(
                tables,
                chals,
                term_offsets,
                term_vars,
                0,
                Q64,
            )
            return out
        return fn

    raise ValueError(f"unsupported bits={bits}")


def build_specialized_call(bits, tables, chals, poly_name):
    poly_id = base.POLY_IDS[poly_name]

    if bits == 32:
        def fn():
            return sumcheck_cuda_ext.sumcheck_hyperplonk_full_mont_u32_cuda(
                tables,
                chals,
                Q32,
                poly_id,
            )
        return fn

    if bits == 64:
        def fn():
            return sumcheck_cuda_ext.sumcheck_hyperplonk_full_mont_u64_cuda(
                tables,
                chals,
                0,
                Q64,
                poly_id,
            )
        return fn

    raise ValueError(f"unsupported bits={bits}")


def fmt_row(values, widths):
    return " | ".join(str(v).rjust(w) for v, w in zip(values, widths))


def fmt_header(values, widths):
    return " | ".join(str(v).center(w) for v, w in zip(values, widths))


def required_symbols(bits):
    if bits == 32:
        return [
            "sumcheck_terms_full_mont_u32_cuda",
            "sumcheck_hyperplonk_full_mont_u32_cuda",
        ]
    if bits == 64:
        return [
            "sumcheck_terms_full_mont_u64_cuda",
            "sumcheck_hyperplonk_full_mont_u64_cuda",
        ]
    raise ValueError(bits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", default="all", help="32, 64, or all")
    ap.add_argument("--num-vars", default="20")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--polys", default="all")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.bits.strip().lower() == "all":
        bits_list = [32, 64]
    else:
        bits_list = [int(x) for x in args.bits.split(",")]

    for bits in bits_list:
        for name in required_symbols(bits):
            if not hasattr(sumcheck_cuda_ext, name):
                raise RuntimeError(f"missing required symbol for bits={bits}: {name}")

    num_vars_list = [int(x) for x in args.num_vars.split(",")]

    if args.polys.strip().lower() == "all":
        poly_names = list(base.DEFAULT_ORDER)
    else:
        poly_names = [x.strip() for x in args.polys.split(",") if x.strip()]

    print("device:", torch.cuda.get_device_name())
    print("SC_EVAL_VARIANT:", os.environ.get("SC_EVAL_VARIANT", "unset/default"))
    print("backend funcs:", [x for x in dir(sumcheck_cuda_ext) if "sumcheck" in x or "montgomery" in x])
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

    widths = [24, 64, 10, 5, 5, 4, 12, 10, 8, 12, 10, 15, 13, 12]

    for bits in bits_list:
        print()
        print(f"bits = {bits}")
        print()

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
                seed = args.seed + bits * 100000 + nv + base.POLY_IDS[poly] * 100

                tables, chals = make_inputs(bits, terms, nv, seed)

                generic_fn = build_generic_call(bits, tables, chals, terms)
                spec_fn = build_specialized_call(bits, tables, chals, poly)

                _gf, gmed, gp90, gout = bench(generic_fn, args.warmup, args.runs)
                _sf, smed, sp90, sout = bench(spec_fn, args.warmup, args.runs)

                if args.check:
                    g = gout.detach().cpu()
                    s = sout.detach().cpu()

                    if not torch.equal(g, s):
                        diff = (g != s).nonzero()[0].tolist()
                        raise AssertionError(
                            f"generic/specialized mismatch bits={bits} "
                            f"poly={poly} nv={nv} first={diff}"
                        )

                n = 1 << nv
                speedup = gmed / smed if smed > 0 else float("inf")
                generic_mpts = (n / gmed) / 1000.0
                spec_mpts = (n / smed) / 1000.0

                function = getattr(base, "FUNCTIONS", {}).get(poly, poly)

                row = [
                    poly,
                    function,
                    f"{n:d}",
                    f"{base.rows_for_terms(terms):d}",
                    f"{len(terms):d}",
                    f"{base.degree_for_terms(terms):d}",
                    f"{gmed:.3f}",
                    f"{smed:.3f}",
                    f"{speedup:.2f}x",
                    f"{gp90:.3f}",
                    f"{sp90:.3f}",
                    f"{generic_mpts:.2f}",
                    f"{spec_mpts:.2f}",
                    str(tuple(gout.shape)),
                ]

                print(fmt_row(row, widths))

            print()


if __name__ == "__main__":
    main()
