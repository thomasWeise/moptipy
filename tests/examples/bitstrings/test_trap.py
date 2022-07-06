"""Test the trap objective function."""
from typing import Final

import numpy as np
from numpy.random import Generator, default_rng

from moptipy.examples.bitstrings.trap import Trap
from moptipy.spaces.bitstrings import BitStrings
from moptipy.tests.objective import validate_objective
from moptipy.tests.on_bitstrings import random_bit_string


def _trap(x: np.ndarray, dim: int) -> int:
    """The comparison implementation of trap."""
    s: int = 0
    for xx in x:
        if xx:
            s += 1
    if (s < 0) or (s > dim):
        raise ValueError(f"x={x}, len(x)={len(x)}, dim={dim}, trap(x)={s}!")
    if s == dim:
        return 0
    return s + 1


def test_trap() -> None:
    """Test the trap objective function."""
    random: Final[Generator] = default_rng()
    space: BitStrings = BitStrings(int(random.integers(2, 32)))
    f: Trap = Trap(space.dimension)

    validate_objective(
        objective=f,
        solution_space=space,
        make_solution_space_element_valid=random_bit_string,
        is_deterministic=True,
        lower_bound_threshold=0,
        upper_bound_threshold=space.dimension,
        must_be_equal_to=lambda xx, dim=space.dimension: _trap(xx, dim))

    space = BitStrings(2)
    f = Trap(2)
    x = space.create()
    for x0 in [True, False]:
        x[0] = x0
        for x1 in [True, False]:
            x[1] = x1
            f1 = f.evaluate(x)
            f2 = _trap(x, 2)
            if f1 != f2:
                raise ValueError(f"trap({x})={f1}, but {f1}!={f2}!")

    space = BitStrings(3)
    f = Trap(3)
    x = space.create()
    for x0 in [True, False]:
        x[0] = x0
        for x1 in [True, False]:
            x[1] = x1
            for x2 in [True, False]:
                x[2] = x2
                f1 = f.evaluate(x)
                f2 = _trap(x, 3)
                if f1 != f2:
                    raise ValueError(f"trap({x})={f1}, but {f1}!={f2}!")

    space = BitStrings(4)
    f = Trap(4)
    x = space.create()
    for x0 in [True, False]:
        x[0] = x0
        for x1 in [True, False]:
            x[1] = x1
            for x2 in [True, False]:
                x[2] = x2
                for x3 in [True, False]:
                    x[3] = x3
                    f1 = f.evaluate(x)
                    f2 = _trap(x, 4)
                    if f1 != f2:
                        raise ValueError(f"trap({x})={f1}, but {f1}!={f2}!")
