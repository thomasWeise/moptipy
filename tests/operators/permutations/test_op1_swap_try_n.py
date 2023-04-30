"""Test the unary swap-try-n operation."""

from typing import Callable, Final

import numpy as np
from numpy.random import Generator, default_rng

from moptipy.operators.permutations.op0_shuffle import Op0Shuffle
from moptipy.operators.permutations.op1_swap_try_n import Op1SwapTryN
from moptipy.operators.tools import exponential_step_size
from moptipy.spaces.permutations import Permutations
from moptipy.tests.on_permutations import (
    validate_op1_with_step_size_on_permutations,
)


def test_op1_swaptn() -> None:
    """Test the unary swap-exactly-n operation."""

    def _min_unique(samples, pwrx) -> int:
        return max(1, min(samples, pwrx.n()) // 2)

    validate_op1_with_step_size_on_permutations(
        Op1SwapTryN, None, _min_unique,
        [0.0, 0.1, 0.2, 0.5, 1.0],
        perm_filter=lambda p: 2 <= p.dimension <= 2000)


def test_op1_swaptn_exact() -> None:
    """Test the exact number of swaps by a swap-try-n op."""
    random: Final[Generator] = default_rng()
    perm: Final[Permutations] = Permutations.standard(
        int(random.integers(10, 100)))
    x1: Final[np.ndarray] = perm.create()
    Op0Shuffle(perm).op0(random, x1)
    x2: Final[np.ndarray] = perm.create()
    op1: Final[Op1SwapTryN] = Op1SwapTryN(perm)
    op1.initialize()
    op: Final[Callable[[Generator, np.ndarray,
                        np.ndarray, float], None]] = op1.op1

    for _ in range(1000):
        steps = random.integers(0, 101) / 100
        assert 0.0 <= steps <= 1.0
        changes = exponential_step_size(steps, 2, len(x1))
        assert 2 <= changes <= len(x1)
        op(random, x2, x1, steps)
        assert sum(x1 != x2) == changes
