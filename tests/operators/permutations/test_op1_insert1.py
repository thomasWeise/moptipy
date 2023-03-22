"""Test the unary insertion operation."""
from typing import Callable

import numpy as np
from numpy.random import default_rng

# noinspection PyProtectedMember
from moptipy.operators.permutations.op1_insert1 import Op1Insert1, _op1_rotate
from moptipy.tests.on_permutations import validate_op1_on_permutations


def test_op1_insert1() -> None:
    """Test the unary insertion operation."""

    def _min_unique(samples, pwr) -> int:
        return max(1, min(samples, pwr.n()) // 2)

    validate_op1_on_permutations(Op1Insert1(), min_unique_samples=_min_unique)


def test_rotate() -> None:
    """Test the rotation operator."""
    rnd: Callable[[int], int] = default_rng().integers
    for i in range(1, 11):
        x: np.ndarray = np.array([rnd(10) for _ in range(i)], int)
        assert len(x) == i

        for j in range(i):
            for k in range(i):
                dst1: np.ndarray = x.copy()
                dst2: np.ndarray = x.copy()
                if j < k:
                    v = dst1[j]
                    dst1[j:k] = dst1[j + 1:k + 1]
                    dst1[k] = v
                elif j > k:
                    v = dst1[j]
                    dst1[k + 1:j + 1] = dst1[k:j]
                    dst1[k] = v

                res = _op1_rotate(dst2, j, k)
                assert res == all(dst1 == x)
                if j == k:
                    assert res
                assert all(dst2 == dst1)
