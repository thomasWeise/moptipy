"""Test the jump objective function."""
from typing import Final

import numpy as np
from numpy.random import Generator, default_rng

from moptipy.examples.bitstrings.jump import Jump
from moptipy.spaces.bitstrings import BitStrings
from moptipy.tests.objective import validate_objective
from moptipy.tests.on_bitstrings import random_bit_string


def _jump(x: np.ndarray, n: int, k: int) -> int:
    """The comparison implementation of jump."""
    s: int = 0
    for xx in x:
        if xx:
            s += 1
    if (s == n) or (s <= (n - k)):
        return n - s
    return k + s


def test_jump() -> None:
    """Test the jump objective function."""
    random: Final[Generator] = default_rng()

    for n in range(3, 8):
        opt = np.ones(n, bool)
        local_worst = np.zeros(n, bool)

        for k in range(2, n >> 1):
            f: Jump = Jump(n, k)
            space: BitStrings = f.space()
            worst = opt.copy()
            worst[random.integers(n)] = False
            almost_worst = opt.copy()
            x1: int = random.integers(n)
            x2: int = x1
            while x1 == x2:
                x2 = random.integers(n)
            almost_worst[x1] = False
            almost_worst[x2] = False
            local_almost_worst = np.zeros(n, bool)
            local_almost_worst[random.integers(n)] = True

            validate_objective(
                objective=f,
                solution_space=space,
                make_solution_space_element_valid=random_bit_string,
                is_deterministic=True,
                lower_bound_threshold=0,
                upper_bound_threshold=n + k - 1,
                must_be_equal_to=lambda x, nn=n, kk=k: _jump(x, nn, kk))

            assert f.evaluate(opt) == 0
            assert f.evaluate(worst) == n + k - 1
            if k > 2:
                assert f.evaluate(almost_worst) == n + k - 2
            else:
                assert f.evaluate(almost_worst) == 2
            assert f.evaluate(local_worst) == n
            assert f.evaluate(local_almost_worst) == n - 1
