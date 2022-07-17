"""Test the priority-based scalarization."""
from typing import Final

import numpy as np
from numpy.random import Generator, default_rng

from moptipy.api.mo_problem import MOSOProblemBridge
from moptipy.examples.bitstrings.onemax import OneMax, onemax
from moptipy.spaces.bitstrings import BitStrings
from moptipy.tests.mo_problem import validate_mo_problem
from moptipy.tests.on_bitstrings import random_bit_string


def test_mosobridge() -> None:
    """Test the mo-so bridge on onemax."""
    random: Final[Generator] = default_rng()
    space: Final[BitStrings] = BitStrings(int(random.integers(2, 32)))
    f1: Final[OneMax] = OneMax(space.dimension)

    def mpet(x: np.ndarray) -> int:
        return onemax(x)

    mo_problem: Final[MOSOProblemBridge] = MOSOProblemBridge(f1)
    validate_mo_problem(
        mo_problem=mo_problem,
        solution_space=space,
        make_solution_space_element_valid=random_bit_string,
        is_deterministic=True,
        lower_bound_threshold=0,
        upper_bound_threshold=space.dimension,
        must_be_equal_to=mpet)
