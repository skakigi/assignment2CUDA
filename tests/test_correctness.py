from __future__ import annotations

import math
import random

import numpy as np

import student
from src.sumcheck_cpu_reference import FIELD_MODULUS, sumcheck_reference


def make_case(num_tables: int, rounds: int, modulus: int):
    rng = random.Random(num_tables * 100 + rounds)
    n = 1 << rounds
    tables = [[rng.randrange(0, min(modulus, 10_000)) for _ in range(n)] for _ in range(num_tables)]
    challenges = [rng.randrange(0, min(modulus, 10_000)) for _ in range(rounds)]
    return tables, challenges


def test_reference_shapes_and_values_small():
    tables, rs = make_case(2, 3, 101)
    out = sumcheck_reference(tables, rs, 101)
    assert len(out) == 3
    assert all(len(row) == 3 for row in out)
    assert all(0 <= x < 101 for row in out for x in row)


def test_student_matches_reference_small_prime():
    for num_tables in [1, 2, 3, 4]:
        for rounds in [1, 2, 5]:
            tables, rs = make_case(num_tables, rounds, 2305843009213693951)
            expected = np.asarray(sumcheck_reference(tables, rs, 2305843009213693951), dtype=np.uint64)
            got = student.sumcheck(np.asarray(tables, dtype=np.uint64), np.asarray(rs, dtype=np.uint64), 2305843009213693951)
            assert np.array_equal(expected, got)


def test_student_matches_reference_goldilocks():
    tables, rs = make_case(3, 4, FIELD_MODULUS)
    expected = np.asarray(sumcheck_reference(tables, rs, FIELD_MODULUS), dtype=np.uint64)
    got = student.sumcheck(np.asarray(tables, dtype=np.uint64), np.asarray(rs, dtype=np.uint64), FIELD_MODULUS)
    assert np.array_equal(expected, got)


