"""Test the priority-based scalarization."""
from typing import Final

import numpy as np
from numpy.random import Generator, default_rng

from moptipy.examples.bitstrings.leadingones import LeadingOnes, leadingones
from moptipy.examples.bitstrings.onemax import OneMax, onemax
from moptipy.mo.problem.weighted_sum import Prioritize
from moptipy.spaces.bitstrings import BitStrings
from moptipy.tests.mo_problem import validate_mo_problem
from moptipy.tests.on_bitstrings import random_bit_string


def test_prioritize_on_onemax_and_leadingones() -> None:
    """Test the weighted sum-based prioritization on onemax, leadingones."""
    random: Final[Generator] = default_rng()
    space: Final[BitStrings] = BitStrings(int(random.integers(2, 32)))
    f1: Final[OneMax] = OneMax(space.dimension)
    f2: Final[LeadingOnes] = LeadingOnes(space.dimension)
    f3: Final[OneMax] = OneMax(space.dimension)

    w3: Final[int] = 1
    w2: Final[int] = (space.dimension + 1)
    w1: Final[int] = w2 * (space.dimension + 1)

    def mpet(x: np.ndarray, ww1=w1, ww2=w2, ww3=w3) -> int:
        return (ww1 * onemax(x)) + (ww2 * leadingones(x)) \
            + (ww3 * onemax(x))

    mo_problem: Final[Prioritize] = Prioritize([f1, f2, f3])
    if mo_problem.weights != (w1, w2, w3):
        raise ValueError(
            f"got weights {mo_problem.weights}, but expected {(w1, w2, w3)}.")

    validate_mo_problem(
        mo_problem=mo_problem,
        solution_space=space,
        make_solution_space_element_valid=random_bit_string,
        is_deterministic=True,
        lower_bound_threshold=0,
        upper_bound_threshold=(w1 + w2 + w3) * space.dimension,
        must_be_equal_to=mpet)
