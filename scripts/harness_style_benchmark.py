#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
import student
from rich.console import Console
from rich.table import Table

try:
    from sumcheck_cpu_reference import sumcheck_reference
except Exception:
    from src.sumcheck_cpu_reference import sumcheck_reference

console = Console()

EXPRS = ["a", "a*b", "a*b + c", "a*b*c"]

MODULI = {
    32: 4294967291,                  # 2^32 - 5
    61: 2305843009213693951,         # 2^61 - 1
    64: 18446744069414584321,        # Goldilocks
}

def parse_csv_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def _u64_to_i64_safe(x: torch.Tensor, q: int) -> torch.Tensor:
    # PyTorch CUDA does not implement many uint64 arithmetic ops.
    # For 32-bit and 61-bit moduli, all values fit in signed int64,
    # so the benchmark wrapper can safely compose expressions in int64.
    if q >= (1 << 63):
        raise NotImplementedError(
            "PyTorch CUDA uint64 arithmetic is unavailable for q >= 2^63 in this wrapper. "
            "Use --bits 32 or --bits 61, or implement a tiny CUDA add/extend kernel."
        )
    return x.to(torch.int64)

def mod_add_small_cuda(x: torch.Tensor, y: torch.Tensor, q: int) -> torch.Tensor:
    xi = _u64_to_i64_safe(x, q)
    yi = _u64_to_i64_safe(y, q)
    s = xi + yi
    s = torch.where(s >= q, s - q, s)
    return s.to(torch.uint32)

def extend_linear_to_quadratic_cuda(vals: torch.Tensor, q: int) -> torch.Tensor:
    # vals has columns [f(0), f(1)].
    # Need f(2) for a degree-2 expression: f(2) = 2*f(1) - f(0).
    vi = _u64_to_i64_safe(vals, q)
    v0 = vi[:, 0]
    v1 = vi[:, 1]

    v2 = (2 * v1 - v0) % q
    out = torch.cat([vi, v2[:, None]], dim=1)
    return out.to(torch.uint32)

def mod_add_cpu_rows(a, b, q: int):
    return [[(int(x) + int(y)) % q for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]

def extend_linear_to_quadratic_cpu(vals, q: int):
    out = []
    for row in vals:
        v0, v1 = int(row[0]), int(row[1])
        v2 = (2 * v1 - v0) % q
        out.append([v0, v1, v2])
    return out

def make_inputs(num_vars: int, bits: int, seed: int):
    q = MODULI[bits]
    n = 1 << num_vars
    torch.manual_seed(seed)

    # Generate values in a signed-safe range, then cast to uint64.
    # For correctness/benchmark structure this is fine.
    high = min(q, 2**31 - 1)

    tables = torch.randint(
        0,
        high,
        (3, n),
        device="cuda",
        dtype=torch.int64,
    ).to(torch.uint32).contiguous()

    challenges = torch.randint(
        0,
        high,
        (num_vars,),
        device="cuda",
        dtype=torch.int64,
    ).to(torch.uint32).contiguous()

    return tables, challenges, q

def compute_expr_cuda(expr: str, tables: torch.Tensor, challenges: torch.Tensor, q: int) -> torch.Tensor:
    # Fused native CUDA path. This avoids composing a*b+c with PyTorch CUDA ops.
    return student.sumcheck_expr(tables, challenges, q, expr)


def _expr_terms(expr: str):
    e = expr.replace(" ", "")
    if e == "a":
        return [[0]]
    if e == "a*b":
        return [[0, 1]]
    if e == "a*b+c":
        return [[0, 1], [2]]
    if e == "a*b*c":
        return [[0, 1, 2]]
    raise ValueError(f"unknown expr: {expr}")

def _mod_add(a: int, b: int, q: int) -> int:
    return (a + b) % q

def _mod_sub(a: int, b: int, q: int) -> int:
    return (a - b) % q

def _mod_mul(a: int, b: int, q: int) -> int:
    return (a * b) % q

def _line_eval_split(zero_eval: int, one_eval: int, t: int, q: int) -> int:
    return (zero_eval + t * ((one_eval - zero_eval) % q)) % q

def sumcheck_reference_split_halves(expr: str, tables, challenges, q: int):
    """CPU reference matching uploaded-base CUDA layout.

    Each round pairs table[i] with table[i + half], not adjacent entries.
    This matches the original assignment-style MLE fold direction.
    """
    terms = _expr_terms(expr)
    degree = max(len(term) for term in terms)
    current = [[int(x) % q for x in row] for row in tables]
    chals = [int(x) % q for x in challenges]
    rounds = (len(current[0]).bit_length() - 1)

    out = []
    for rnd in range(rounds):
        length = len(current[0])
        half = length // 2

        row = []
        for t in range(degree + 1):
            acc = 0
            for i in range(half):
                expr_val = 0
                for term in terms:
                    prod = 1
                    for var_idx in term:
                        z = current[var_idx][i]
                        o = current[var_idx][i + half]
                        prod = (prod * _line_eval_split(z, o, t, q)) % q
                    expr_val = (expr_val + prod) % q
                acc = (acc + expr_val) % q
            row.append(acc)
        out.append(row)

        # Uploaded-base/original-style protocol has num_rounds-1 prover challenges.
        if rnd < rounds - 1:
            r = chals[rnd]
            next_tables = []
            for table in current:
                next_tables.append([
                    _line_eval_split(table[i], table[i + half], r, q)
                    for i in range(half)
                ])
            current = next_tables

    return out



def compute_expr_cpu(expr: str, tables: torch.Tensor, challenges: torch.Tensor, q: int):
    tables_cpu = tables.cpu().numpy().tolist()
    ch_cpu = challenges.cpu().numpy().tolist()
    return sumcheck_reference_split_halves(expr, tables_cpu, ch_cpu, q)

def time_cuda_call(fn):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    out = fn()
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end), out

def check_correctness(expr: str, tables: torch.Tensor, challenges: torch.Tensor, q: int, out: torch.Tensor):
    ref = compute_expr_cpu(expr, tables, challenges, q)
    out_cpu = out.detach().cpu()
    ref_t = torch.tensor(ref, dtype=out_cpu.dtype)

    if not torch.equal(out_cpu, ref_t):
        mismatch = (out_cpu != ref_t).nonzero()
        first = mismatch[0].tolist() if mismatch.numel() else None
        raise AssertionError(
            f"correctness mismatch expr={expr}, first mismatch={first}, "
            f"out_shape={tuple(out_cpu.shape)}, ref_shape={tuple(ref_t.shape)}"
        )

def summarize_case(testcase: str, bits: int, n: int, expr: str, compile_ms: float, timed_ms: list[float]):
    med = statistics.median(timed_ms)
    p90 = statistics.quantiles(timed_ms, n=10)[8] if len(timed_ms) >= 10 else max(timed_ms)
    mpts = n / (med * 1000.0) if med > 0 else 0.0
    return {
        "testcase": testcase,
        "bits": bits,
        "N": n,
        "expr": expr,
        "compile_ms": compile_ms,
        "median_ms": med,
        "p90_ms": p90,
        "mpts": mpts,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-vars", default="4", help="Comma-separated, e.g. 4,16,20")
    ap.add_argument("--bits", type=int, choices=sorted(MODULI), default=32)
    ap.add_argument("--cases", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--check-up-to", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")

    status = student.native_status()
    if "OK" not in status:
        raise RuntimeError(f"native extension is unavailable: {status}")

    latency_rows = []
    invocation_rows = []

    for num_vars in parse_csv_ints(args.num_vars):
        n = 1 << num_vars

        for case_idx in range(args.cases):
            testcase = f"v{num_vars}_case{args.bits}_{case_idx}"
            seed = args.seed + 100000 * num_vars + case_idx
            tables, challenges, q = make_inputs(num_vars, args.bits, seed)

            for expr in EXPRS:
                # Match the original table's "compile" phase name.
                # For this native CUDA version this is really first-call/setup latency,
                # because extension compilation happened earlier in build_extension.py.
                compile_ms, out = time_cuda_call(lambda: compute_expr_cuda(expr, tables, challenges, q))
                invocation_rows.append((testcase, expr, "compile", 0, compile_ms))

                for i in range(1, args.warmup + 1):
                    ms, out = time_cuda_call(lambda: compute_expr_cuda(expr, tables, challenges, q))
                    invocation_rows.append((testcase, expr, "warmup", i, ms))

                if num_vars <= args.check_up_to:
                    check_correctness(expr, tables, challenges, q, out)

                timed = []
                for i in range(1, args.runs + 1):
                    ms, out = time_cuda_call(lambda: compute_expr_cuda(expr, tables, challenges, q))
                    timed.append(ms)
                    invocation_rows.append((testcase, expr, "timed", i, ms))

                latency_rows.append(summarize_case(testcase, args.bits, n, expr, compile_ms, timed))

    summary = Table(title="Sumcheck Latency")
    summary.add_column("testcase")
    summary.add_column("bits", justify="right")
    summary.add_column("N", justify="right")
    summary.add_column("expr")
    summary.add_column("compile\n(ms)", justify="right")
    summary.add_column("median\n(ms)", justify="right")
    summary.add_column("p90 (ms)", justify="right")
    summary.add_column("Mpts/s", justify="right")

    for r in latency_rows:
        summary.add_row(
            r["testcase"],
            str(r["bits"]),
            str(r["N"]),
            r["expr"],
            f"{r['compile_ms']:.3f}",
            f"{r['median_ms']:.3f}",
            f"{r['p90_ms']:.3f}",
            f"{r['mpts']:.2f}",
        )

    detail = Table(title="Per-Invocation Times")
    detail.add_column("testcase")
    detail.add_column("expr")
    detail.add_column("phase")
    detail.add_column("iter", justify="right")
    detail.add_column("elapsed (ms)", justify="right")

    for testcase, expr, phase, it, ms in invocation_rows:
        detail.add_row(testcase, expr, phase, str(it), f"{ms:.3f}")

    console.print(summary)
    console.print(detail)
    console.print(f"Device: gpu ({torch.cuda.get_device_name(0)})")

if __name__ == "__main__":
    main()
