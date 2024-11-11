"""Test the N-Queens objective function."""
from typing import Final

from numpy.random import Generator, default_rng

from moptipy.examples.bitstrings.nqueens import NQueens
from moptipy.spaces.bitstrings import BitStrings
from moptipy.tests.objective import validate_objective
from moptipy.tests.on_bitstrings import random_bit_string


def test_nqueens() -> None:
    """Test the N-Queens objective function."""
    random: Final[Generator] = default_rng()
    space: BitStrings = BitStrings(int(random.integers(4, 12)) ** 2)
    f: NQueens = NQueens(space.dimension)

    validate_objective(
        objective=f,
        solution_space=space,
        make_solution_space_element_valid=random_bit_string,
        is_deterministic=True,
        lower_bound_threshold=0)
