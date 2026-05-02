#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path

import numpy as np

from src.sumcheck_cpu_reference import FIELD_MODULUS, sumcheck_reference


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("test_inputs"))
    ap.add_argument("--num-tables", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--modulus", type=int, default=FIELD_MODULUS)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)
    n = 1 << args.rounds
    hi = min(args.modulus, 2**31 - 1)
    tables = rng.integers(0, hi, size=(args.num_tables, n), dtype=np.uint64)
    challenges = rng.integers(0, hi, size=(args.rounds,), dtype=np.uint64)
    expected = np.asarray(sumcheck_reference(tables.tolist(), challenges.tolist(), args.modulus), dtype=np.uint64)

    np.save(args.out / "eval_tables.npy", tables)
    np.save(args.out / "challenges.npy", challenges)
    np.save(args.out / "expected.npy", expected)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
