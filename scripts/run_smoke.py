#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import random

import numpy as np

import student
from src.sumcheck_cpu_reference import FIELD_MODULUS, sumcheck_reference


def make_case(num_tables: int, rounds: int, modulus: int):
    n = 1 << rounds
    rng = random.Random(0)
    tables = [[rng.randrange(0, min(modulus, 10_000)) for _ in range(n)] for _ in range(num_tables)]
    challenges = [rng.randrange(0, min(modulus, 10_000)) for _ in range(rounds)]
    return tables, challenges


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", action="store_true", help="also run native CUDA path with torch.uint64 tensors")
    ap.add_argument("--modulus", type=int, default=FIELD_MODULUS)
    ap.add_argument("--num-tables", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=5)
    args = ap.parse_args()

    tables, challenges = make_case(args.num_tables, args.rounds, args.modulus)
    ref = sumcheck_reference(tables, challenges, args.modulus)
    got = student.sumcheck(np.asarray(tables, dtype=np.uint64), np.asarray(challenges, dtype=np.uint64), args.modulus)
    assert np.array_equal(np.asarray(ref, dtype=np.uint64), np.asarray(got, dtype=np.uint64)), "CPU fallback mismatch"
    print("CPU/reference smoke OK", np.asarray(got, dtype=np.uint64).shape)

    if args.cuda:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("--cuda requested but torch.cuda.is_available() is False")
        t = torch.tensor(tables, dtype=torch.uint64, device="cuda")
        r = torch.tensor(challenges, dtype=torch.uint64, device="cuda")
        out = student.sumcheck(t, r, args.modulus)
        out_cpu = out.detach().cpu().numpy()
        assert np.array_equal(np.asarray(ref, dtype=np.uint64), out_cpu), "CUDA mismatch"
        print("CUDA smoke OK", tuple(out.shape))
        print(student.native_status())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
