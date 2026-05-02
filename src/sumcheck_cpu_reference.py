"""Readable CPU reference for product-of-MLE SumCheck.

The table layout binds the least-significant variable first by adjacent pairs:
    (0,1), (2,3), ...

For k input tables, the round polynomial has degree <= k and is emitted as
its evaluations at x = 0, 1, ..., k.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence

FIELD_MODULUS = 18446744069414584321  # Goldilocks: 2^64 - 2^32 + 1


def _as_python_matrix(eval_tables: Iterable[Iterable[int]]) -> List[List[int]]:
    return [[int(x) for x in row] for row in eval_tables]


def _as_python_vector(challenges: Iterable[int]) -> List[int]:
    return [int(x) for x in challenges]


def _is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def _validate(tables: Sequence[Sequence[int]], challenges: Sequence[int]) -> None:
    if not tables:
        raise ValueError("eval_tables must contain at least one table")
    n = len(tables[0])
    if not _is_power_of_two(n):
        raise ValueError(f"table length must be a power of two, got {n}")
    for i, row in enumerate(tables):
        if len(row) != n:
            raise ValueError(f"all tables must have equal length; row 0 has {n}, row {i} has {len(row)}")
    rounds = n.bit_length() - 1
    if len(challenges) != rounds:
        raise ValueError(f"expected {rounds} challenges for table length {n}, got {len(challenges)}")


def sumcheck_reference(eval_tables, challenges, modulus: int = FIELD_MODULUS):
    """Return SumCheck round evaluations for a product of MLE tables.

    Args:
        eval_tables: shape (num_tables, 2**rounds), values modulo modulus.
        challenges: shape (rounds,), verifier challenges used for MLE updates.
        modulus: prime field modulus. The reference works for any positive modulus.

    Returns:
        A list of lists with shape (rounds, num_tables + 1). Row r stores the
        round-polynomial evaluations at x = 0, 1, ..., num_tables.
    """
    if modulus <= 1:
        raise ValueError("modulus must be > 1")

    tables = _as_python_matrix(eval_tables)
    rs = [r % modulus for r in _as_python_vector(challenges)]
    _validate(tables, rs)

    num_tables = len(tables)
    degree = num_tables
    cur_len = len(tables[0])
    transcript = []

    for r in rs:
        pair_count = cur_len // 2
        round_evals = [0] * (degree + 1)

        for pair in range(pair_count):
            base = 2 * pair
            # Evaluate prod_j (a_j + x * (b_j - a_j)) at x = 0..degree.
            for x in range(degree + 1):
                prod = 1
                for t in range(num_tables):
                    a = tables[t][base] % modulus
                    b = tables[t][base + 1] % modulus
                    val = (a + x * ((b - a) % modulus)) % modulus
                    prod = (prod * val) % modulus
                round_evals[x] = (round_evals[x] + prod) % modulus

        transcript.append(round_evals)

        # Bind the current variable to the verifier challenge.
        next_tables = [[0] * pair_count for _ in range(num_tables)]
        for t in range(num_tables):
            row = tables[t]
            out = next_tables[t]
            for pair in range(pair_count):
                a = row[2 * pair] % modulus
                b = row[2 * pair + 1] % modulus
                out[pair] = (a + r * ((b - a) % modulus)) % modulus
        tables = next_tables
        cur_len = pair_count

    return transcript
