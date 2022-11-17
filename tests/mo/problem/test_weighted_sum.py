"""Test the weighted sum scalarization."""
from typing import Final

import numpy as np
from numpy.random import Generator, default_rng

from moptipy.examples.bitstrings.leadingones import LeadingOnes, leadingones
from moptipy.examples.bitstrings.onemax import OneMax, onemax
from moptipy.mo.problem.weighted_sum import WeightedSum
from moptipy.spaces.bitstrings import BitStrings
from moptipy.tests.mo_problem import validate_mo_problem
from moptipy.tests.on_bitstrings import random_bit_string


def test_weighted_sum_on_onemax_and_leadingones() -> None:
    """Test the weighted sum method on onemax and leadingones."""
    random: Final[Generator] = default_rng()
    space: Final[BitStrings] = BitStrings(int(random.integers(2, 32)))
    f1: Final[OneMax] = OneMax(space.dimension)
    f2: Final[LeadingOnes] = LeadingOnes(space.dimension)

    w1: Final[int] = int(random.integers(1, 100))
    w2: Final[int] = int(random.integers(1, 100))

    weights: list[int | float] = [
        w1 if random.integers(2) <= 0 else float(w1),
        w2 if random.integers(2) <= 0 else float(w2),
    ]

    def mpet(x: np.ndarray, ww1=w1, ww2=w2) -> int:
        return (ww1 * onemax(x)) + (ww2 * leadingones(x))

    mo_problem: Final[WeightedSum] = WeightedSum([f1, f2], weights)
    if mo_problem.weights != (w1, w2):
        raise ValueError(
            f"got weights {mo_problem.weights}, but expected {(w1, w2)}.")

    validate_mo_problem(
        mo_problem=mo_problem,
        solution_space=space,
        make_solution_space_element_valid=random_bit_string,
        is_deterministic=True,
        lower_bound_threshold=0,
        upper_bound_threshold=(w1 + w2) * space.dimension,
        must_be_equal_to=mpet)
