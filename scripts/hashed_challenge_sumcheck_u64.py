import argparse
import hashlib
import math
import statistics
import sys

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import sumcheck_cuda_ext
import paper_poly_compare as base

Q64 = (1 << 64) - (1 << 32) + 1


def flatten_terms(terms):
    offsets = [0]
    flat = []
    for term in terms:
        flat.extend(term)
        offsets.append(len(flat))
    return offsets, flat


def make_inputs(poly, num_vars, seed):
    terms = base.POLYS[poly]
    rows = base.rows_for_terms(terms)
    n = 1 << num_vars

    rng = np.random.default_rng(seed)
    tables_np = rng.integers(0, 1000, size=(rows, n), dtype=np.uint64)

    return torch.from_numpy(tables_np).cuda().contiguous()


def round_evals_to_bytes(round_evals_cpu):
    arr = np.asarray(round_evals_cpu, dtype=np.uint64)
    return arr.astype("<u8", copy=False).tobytes()


def derive_challenge(transcript: bytes, round_idx: int, round_evals_cpu):
    h = hashlib.sha3_256()
    h.update(b"sumcheck-u64-transcript-v1")
    h.update(round_idx.to_bytes(4, "little"))
    h.update(transcript)
    h.update(round_evals_to_bytes(round_evals_cpu))
    digest = h.digest()

    challenge = int.from_bytes(digest, "little") % Q64
    if challenge == Q64:
        challenge = 0

    next_transcript = hashlib.sha3_256(
        transcript
        + round_idx.to_bytes(4, "little")
        + round_evals_to_bytes(round_evals_cpu)
        + challenge.to_bytes(8, "little")
    ).digest()

    return challenge, next_transcript


def time_call(fn):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), out


def run_hashed_backend(poly, num_vars, seed, backend, check=True):
    terms = base.POLYS[poly]
    offsets, flat = flatten_terms(terms)

    term_offsets = torch.tensor(offsets, dtype=torch.int32)
    term_vars = torch.tensor(flat, dtype=torch.int32)
    poly_id = int(base.POLY_IDS[poly])

    current = make_inputs(poly, num_vars, seed)
    original = current.clone()

    transcript = hashlib.sha3_256(
        b"poly="
        + poly.encode()
        + b"|num_vars="
        + str(num_vars).encode()
    ).digest()

    round_rows = []
    challenges = []

    total_eval_ms = 0.0
    total_fold_ms = 0.0

    for round_idx in range(num_vars):
        if backend == "generic":
            eval_fn = lambda: sumcheck_cuda_ext.sumcheck_terms_full_mont_u64_round_eval_cuda(
                current,
                term_offsets,
                term_vars,
                0,
                Q64,
            )
        elif backend == "specialized":
            eval_fn = lambda: sumcheck_cuda_ext.sumcheck_hyperplonk_full_mont_u64_round_eval_cuda(
                current,
                0,
                Q64,
                poly_id,
            )
        else:
            raise ValueError(f"unknown backend {backend!r}")

        eval_ms, round_evals = time_call(eval_fn)
        total_eval_ms += eval_ms

        round_cpu = round_evals.detach().cpu().numpy()
        challenge, transcript = derive_challenge(transcript, round_idx, round_cpu)

        round_rows.append(round_evals)
        challenges.append(challenge)

        chal_tensor = torch.tensor([challenge], dtype=torch.uint64, device="cuda")

        if round_idx + 1 < num_vars:
            fold_ms, current = time_call(
                lambda: sumcheck_cuda_ext.fold_full_mont_u64_cuda(
                    current,
                    chal_tensor,
                    0,
                    Q64,
                )
            )
            total_fold_ms += fold_ms

    out = torch.stack(round_rows, dim=0)
    chals = torch.tensor(challenges, dtype=torch.uint64, device="cuda")

    if check:
        if backend == "generic":
            _claim, ref = sumcheck_cuda_ext.sumcheck_terms_full_mont_u64_cuda(
                original,
                chals,
                term_offsets,
                term_vars,
                0,
                Q64,
            )
        else:
            ref = sumcheck_cuda_ext.sumcheck_hyperplonk_full_mont_u64_cuda(
                original,
                chals,
                0,
                Q64,
                poly_id,
            )

        if not torch.equal(out.detach().cpu(), ref.detach().cpu()):
            diff = (out.detach().cpu() != ref.detach().cpu()).nonzero()[0].tolist()
            raise AssertionError(
                f"hashed-vs-full mismatch backend={backend} poly={poly} nv={num_vars} first={diff}"
            )

    return {
        "out": out,
        "challenges": challenges,
        "eval_ms": total_eval_ms,
        "fold_ms": total_fold_ms,
        "total_ms": total_eval_ms + total_fold_ms,
        "transcript": transcript.hex(),
    }


def percentile_nearest_rank(values, percentile):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    idx = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[idx]


def bench_backend(poly, num_vars, seed, warmup, runs, check, backend):
    for _ in range(warmup):
        run_hashed_backend(poly, num_vars, seed, backend=backend, check=check)

    times = []
    last = None
    for _ in range(runs):
        result = run_hashed_backend(poly, num_vars, seed, backend=backend, check=check)
        times.append(result["total_ms"])
        last = result

    return {
        "median_ms": statistics.median(times),
        "p90_ms": percentile_nearest_rank(times, 0.90),
        "last": last,
    }


def bench(poly, num_vars, seed, warmup, runs, check):
    generic = bench_backend(poly, num_vars, seed, warmup, runs, check, "generic")
    spec = bench_backend(poly, num_vars, seed, warmup, runs, check, "specialized")

    g_last = generic["last"]
    s_last = spec["last"]

    if not torch.equal(g_last["out"].detach().cpu(), s_last["out"].detach().cpu()):
        diff = (g_last["out"].detach().cpu() != s_last["out"].detach().cpu()).nonzero()[0].tolist()
        raise AssertionError(
            f"generic-vs-specialized hashed output mismatch poly={poly} nv={num_vars} first={diff}"
        )

    if g_last["challenges"] != s_last["challenges"]:
        raise AssertionError(f"generic-vs-specialized challenge mismatch poly={poly} nv={num_vars}")

    if g_last["transcript"] != s_last["transcript"]:
        raise AssertionError(f"generic-vs-specialized transcript mismatch poly={poly} nv={num_vars}")

    return {
        "generic_ms": generic["median_ms"],
        "spec_ms": spec["median_ms"],
        "speedup": generic["median_ms"] / spec["median_ms"] if spec["median_ms"] > 0 else 0.0,
        "generic_p90": generic["p90_ms"],
        "spec_p90": spec["p90_ms"],
        "last": g_last,
    }


def fmt_row(values, widths):
    return " | ".join(str(v).rjust(w) for v, w in zip(values, widths))


def fmt_header(values, widths):
    return " | ".join(str(v).center(w) for v, w in zip(values, widths))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-vars", default="16")
    ap.add_argument("--polys", default="baseline_mul_add")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    required = [
        "sumcheck_terms_full_mont_u64_round_eval_cuda",
        "sumcheck_hyperplonk_full_mont_u64_round_eval_cuda",
        "fold_full_mont_u64_cuda",
        "sumcheck_terms_full_mont_u64_cuda",
        "sumcheck_hyperplonk_full_mont_u64_cuda",
    ]
    for name in required:
        if not hasattr(sumcheck_cuda_ext, name):
            raise RuntimeError(f"missing required entrypoint: {name}")

    num_vars_list = [int(x) for x in args.num_vars.split(",")]

    if args.polys.strip().lower() == "all":
        poly_names = list(base.DEFAULT_ORDER)
    else:
        poly_names = [x.strip() for x in args.polys.split(",") if x.strip()]

    print("device:", torch.cuda.get_device_name())
    print("challenge mode: SHA3-256 transcript between SumCheck rounds")
    print("backend: u64 full Montgomery generic/spec round eval + fold")
    print()

    headers = [
        "poly",
        "function",
        "num_vars",
        "N",
        "deg",
        "generic_ms",
        "spec_ms",
        "speedup",
        "generic_p90",
        "spec_p90",
        "sha3_prefix",
    ]
    widths = [24, 64, 8, 10, 4, 12, 10, 8, 12, 10, 18]

    for nv in num_vars_list:
        print()
        print(f"num_vars = {nv}")
        print()
        print(fmt_header(headers, widths))
        print("-+-".join("-" * w for w in widths))

        for poly in poly_names:
            terms = base.POLYS[poly]
            result = bench(
                poly=poly,
                num_vars=nv,
                seed=args.seed + nv + base.POLY_IDS[poly] * 100,
                warmup=args.warmup,
                runs=args.runs,
                check=args.check,
            )

            last = result["last"]
            function = getattr(base, "FUNCTIONS", {}).get(poly, poly)

            row = [
                poly,
                function,
                f"{nv:d}",
                f"{1 << nv:d}",
                f"{base.degree_for_terms(terms):d}",
                f"{result['generic_ms']:.3f}",
                f"{result['spec_ms']:.3f}",
                f"{result['speedup']:.2f}x",
                f"{result['generic_p90']:.3f}",
                f"{result['spec_p90']:.3f}",
                last["transcript"][:16],
            ]
            print(fmt_row(row, widths))

        print()


if __name__ == "__main__":
    main()
