"""Test the leadingones objective function."""
from typing import Final

import numpy as np
from numpy.random import Generator, default_rng

from moptipy.examples.bitstrings.leadingones import LeadingOnes
from moptipy.spaces.bitstrings import BitStrings
from moptipy.tests.objective import validate_objective
from moptipy.tests.on_bitstrings import random_bit_string


def _leadingones(x: np.ndarray, dim: int) -> int:
    """The comparison implementation of leadingones."""
    s: int = 0
    for xx in x:
        if not xx:
            break
        s += 1
    if (s < 0) or (s > dim):
        raise ValueError(
            f"x={x}, len(x)={len(x)}, dim={dim}, leadingones(x)={s}!")
    return dim - s


def test_leadingones() -> None:
    """Test the leadingones objective function."""
    random: Final[Generator] = default_rng()
    space: BitStrings = BitStrings(int(random.integers(2, 32)))
    f: LeadingOnes = LeadingOnes(space.dimension)

    validate_objective(
        objective=f,
        solution_space=space,
        make_solution_space_element_valid=random_bit_string,
        is_deterministic=True,
        lower_bound_threshold=0,
        upper_bound_threshold=space.dimension,
        must_be_equal_to=lambda xx, dim=space.dimension:
        _leadingones(xx, dim))

    space = BitStrings(2)
    f = LeadingOnes(2)
    x = space.create()
    for x0 in [True, False]:
        x[0] = x0
        for x1 in [True, False]:
            x[1] = x1
            f1 = f.evaluate(x)
            f2 = _leadingones(x, 2)
            if f1 != f2:
                raise ValueError(f"leadingones({x})={f1}, but {f1}!={f2}!")

    space = BitStrings(3)
    f = LeadingOnes(3)
    x = space.create()
    for x0 in [True, False]:
        x[0] = x0
        for x1 in [True, False]:
            x[1] = x1
            for x2 in [True, False]:
                x[2] = x2
                f1 = f.evaluate(x)
                f2 = _leadingones(x, 3)
                if f1 != f2:
                    raise ValueError(
                        f"leadingones({x})={f1}, but {f1}!={f2}!")

    space = BitStrings(4)
    f = LeadingOnes(4)
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
                    f2 = _leadingones(x, 4)
                    if f1 != f2:
                        raise ValueError(
                            f"leadingones({x})={f1}, but {f1}!={f2}!")
