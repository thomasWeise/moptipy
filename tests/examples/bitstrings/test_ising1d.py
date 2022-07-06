"""Test the ising1d objective function."""
from typing import Final

import numpy as np
from numpy.random import Generator, default_rng

from moptipy.examples.bitstrings.ising1d import Ising1d
from moptipy.spaces.bitstrings import BitStrings
from moptipy.tests.objective import validate_objective
from moptipy.tests.on_bitstrings import random_bit_string


def _ising1d(x: np.ndarray, dim: int) -> int:
    """The comparison implementation of ising1d."""
    s: int = 0
    for i, xx in enumerate(x):
        if x[(i + 1) % dim] == xx:
            s += 1

    if (s < 0) or (s > dim):
        raise ValueError(f"x={x}, len(x)={len(x)}, dim={dim}, ising1d(x)={s}!")
    return dim - s


def test_ising1d() -> None:
    """Test the ising1d objective function."""
    random: Final[Generator] = default_rng()
    space: BitStrings = BitStrings(int(random.integers(2, 32)))
    f: Ising1d = Ising1d(space.dimension)

    validate_objective(
        objective=f,
        solution_space=space,
        make_solution_space_element_valid=random_bit_string,
        is_deterministic=True,
        lower_bound_threshold=0,
        upper_bound_threshold=space.dimension,
        must_be_equal_to=lambda xx, dim=space.dimension: _ising1d(xx, dim))

    space = BitStrings(2)
    f = Ising1d(2)
    x = space.create()
    for x0 in [True, False]:
        x[0] = x0
        for x1 in [True, False]:
            x[1] = x1
            f1 = f.evaluate(x)
            f2 = _ising1d(x, 2)
            if f1 != f2:
                raise ValueError(f"ising1d({x})={f1}, but {f1}!={f2}!")

    space = BitStrings(3)
    f = Ising1d(3)
    x = space.create()
    for x0 in [True, False]:
        x[0] = x0
        for x1 in [True, False]:
            x[1] = x1
            for x2 in [True, False]:
                x[2] = x2
                f1 = f.evaluate(x)
                f2 = _ising1d(x, 3)
                if f1 != f2:
                    raise ValueError(f"ising1d({x})={f1}, but {f1}!={f2}!")

    space = BitStrings(4)
    f = Ising1d(4)
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
                    f2 = _ising1d(x, 4)
                    if f1 != f2:
                        raise ValueError(f"ising1d({x})={f1}, but {f1}!={f2}!")
