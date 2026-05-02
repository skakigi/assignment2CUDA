#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
import student
from src.sumcheck_cpu_reference import sumcheck_reference

MODULI = {
    "u32": 4294967291,                 # 2^32 - 5
    "mersenne61": 2305843009213693951, # 2^61 - 1
    "goldilocks": 18446744069414584321 # 2^64 - 2^32 + 1
}

def parse_csv_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def make_inputs(num_tables: int, num_vars: int, q: int, seed: int):
    torch.manual_seed(seed)
    n = 1 << num_vars

    # Keep random generation in a signed-int-safe range, then cast to uint64.
    # This does not affect benchmark structure; it avoids PyTorch randint edge cases.
    high = min(q, 2**31 - 1)

    tables = torch.randint(
        0,
        high,
        (num_tables, n),
        device="cuda",
        dtype=torch.int64,
    ).to(torch.uint64)

    challenges = torch.randint(
        0,
        high,
        (num_vars,),
        device="cuda",
        dtype=torch.int64,
    ).to(torch.uint64)

    return tables.contiguous(), challenges.contiguous()

def check_correctness(tables: torch.Tensor, challenges: torch.Tensor, q: int, out: torch.Tensor):
    tables_cpu = tables.cpu().numpy().tolist()
    challenges_cpu = challenges.cpu().numpy().tolist()
    ref = sumcheck_reference(tables_cpu, challenges_cpu, q)
    ref_t = torch.tensor(ref, dtype=torch.uint64)
    out_cpu = out.detach().cpu()

    if not torch.equal(out_cpu, ref_t):
        # Print a small diagnostic before failing.
        mismatch = (out_cpu != ref_t).nonzero()
        first = mismatch[0].tolist() if mismatch.numel() else None
        raise AssertionError(
            f"correctness mismatch; first mismatch index={first}, "
            f"out_shape={tuple(out_cpu.shape)}, ref_shape={tuple(ref_t.shape)}"
        )

def bench_one(
    *,
    num_vars: int,
    num_tables: int,
    q: int,
    runs: int,
    warmup: int,
    seed: int,
    check: bool,
):
    tables, challenges = make_inputs(num_tables, num_vars, q, seed)

    status = student.native_status()
    if "OK" not in status:
        raise RuntimeError(
            f"Native CUDA extension is not available: {status}\n"
            "Run source env_cuda.sh and rebuild before benchmarking."
        )

    # Warmup
    out = None
    for _ in range(warmup):
        out = student.sumcheck(tables, challenges, q)
    torch.cuda.synchronize()

    if out is None or not isinstance(out, torch.Tensor) or not out.is_cuda:
        raise RuntimeError(
            "student.sumcheck did not return a CUDA torch.Tensor; "
            "benchmark would not be measuring the native CUDA path."
        )

    expected_shape = (num_vars, num_tables + 1)
    if tuple(out.shape) != expected_shape:
        raise RuntimeError(f"unexpected output shape {tuple(out.shape)}, expected {expected_shape}")

    if check:
        check_correctness(tables, challenges, q, out)

    # Timed runs: use both CUDA events and wall time.
    event_ms = []
    wall_ms = []

    for _ in range(runs):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        t0 = time.perf_counter()
        start_event.record()
        out = student.sumcheck(tables, challenges, q)
        end_event.record()
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        event_ms.append(start_event.elapsed_time(end_event))
        wall_ms.append((t1 - t0) * 1000.0)

    n = 1 << num_vars
    input_mb = tables.numel() * tables.element_size() / (1024**2)
    output_kb = out.numel() * out.element_size() / 1024

    return {
        "num_vars": num_vars,
        "table_len": n,
        "num_tables": num_tables,
        "modulus": q,
        "input_mb": input_mb,
        "output_kb": output_kb,
        "runs": runs,
        "warmup": warmup,
        "event_ms_min": min(event_ms),
        "event_ms_median": statistics.median(event_ms),
        "event_ms_mean": statistics.mean(event_ms),
        "wall_ms_min": min(wall_ms),
        "wall_ms_median": statistics.median(wall_ms),
        "wall_ms_mean": statistics.mean(wall_ms),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-vars", default="4,16,20", help="Comma-separated sweep, e.g. 4,16,20")
    ap.add_argument("--num-tables", type=int, default=3, help="Number of MLE tables; degree=num_tables")
    ap.add_argument("--modulus", choices=sorted(MODULI), default="mersenne61")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check-up-to", type=int, default=16, help="CPU correctness check for num_vars <= this")
    ap.add_argument("--csv", default="", help="Optional CSV output path")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")

    print("native:", student.native_status())
    print("torch:", torch.__version__)
    print("torch CUDA:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    print()

    q = MODULI[args.modulus]
    rows = []

    for nv in parse_csv_ints(args.num_vars):
        do_check = nv <= args.check_up_to
        print(f"case: num_vars={nv}, table_len=2^{nv}, num_tables={args.num_tables}, check={do_check}")
        row = bench_one(
            num_vars=nv,
            num_tables=args.num_tables,
            q=q,
            runs=args.runs,
            warmup=args.warmup,
            seed=args.seed + nv,
            check=do_check,
        )
        rows.append(row)

        print(
            f"  input={row['input_mb']:.2f} MiB, "
            f"event median={row['event_ms_median']:.4f} ms, "
            f"event mean={row['event_ms_mean']:.4f} ms, "
            f"wall median={row['wall_ms_median']:.4f} ms"
        )
        print()

    if args.csv:
        path = Path(args.csv)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path}")

if __name__ == "__main__":
    main()
