import argparse
import hashlib
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


def run_hashed(poly, num_vars, seed, check=True):
    terms = base.POLYS[poly]
    offsets, flat = flatten_terms(terms)

    term_offsets = torch.tensor(offsets, dtype=torch.int32)
    term_vars = torch.tensor(flat, dtype=torch.int32)

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
        eval_ms, round_evals = time_call(
            lambda: sumcheck_cuda_ext.sumcheck_terms_full_mont_u64_round_eval_cuda(
                current,
                term_offsets,
                term_vars,
                0,
                Q64,
            )
        )
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
        _claim, ref = sumcheck_cuda_ext.sumcheck_terms_full_mont_u64_cuda(
            original,
            chals,
            term_offsets,
            term_vars,
            0,
            Q64,
        )

        if not torch.equal(out.detach().cpu(), ref.detach().cpu()):
            diff = (out.detach().cpu() != ref.detach().cpu()).nonzero()[0].tolist()
            raise AssertionError(
                f"hashed-vs-full mismatch poly={poly} nv={num_vars} first={diff}"
            )

    return {
        "out": out,
        "challenges": challenges,
        "eval_ms": total_eval_ms,
        "fold_ms": total_fold_ms,
        "total_ms": total_eval_ms + total_fold_ms,
        "transcript": transcript.hex(),
    }


def bench(poly, num_vars, seed, warmup, runs, check):
    for _ in range(warmup):
        run_hashed(poly, num_vars, seed, check=check)

    times = []
    last = None
    for _ in range(runs):
        result = run_hashed(poly, num_vars, seed, check=check)
        times.append(result["total_ms"])
        last = result

    return {
        "median_ms": statistics.median(times),
        "p90_ms": sorted(times)[int(0.9 * (len(times) - 1))],
        "last": last,
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

    if not hasattr(sumcheck_cuda_ext, "sumcheck_terms_full_mont_u64_round_eval_cuda"):
        raise RuntimeError("missing round eval entrypoint")
    if not hasattr(sumcheck_cuda_ext, "fold_full_mont_u64_cuda"):
        raise RuntimeError("missing fold entrypoint")

    num_vars_list = [int(x) for x in args.num_vars.split(",")]

    if args.polys.strip().lower() == "all":
        poly_names = list(base.DEFAULT_ORDER)
    else:
        poly_names = [x.strip() for x in args.polys.split(",") if x.strip()]

    print("device:", torch.cuda.get_device_name())
    print("challenge mode: SHA3-256 transcript between SumCheck rounds")
    print("backend: u64 full Montgomery round eval + fold")
    print()

    headers = [
        "poly",
        "function",
        "num_vars",
        "N",
        "deg",
        "median_ms",
        "p90_ms",
        "first_chal",
        "transcript_prefix",
    ]
    widths = [24, 64, 8, 10, 4, 10, 10, 20, 18]

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
                f"{result['median_ms']:.3f}",
                f"{result['p90_ms']:.3f}",
                str(last["challenges"][0]),
                last["transcript"][:16],
            ]
            print(fmt_row(row, widths))

        print()


if __name__ == "__main__":
    main()
