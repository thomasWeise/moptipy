"""Test the twomax objective function."""
from typing import Final

from numpy.random import Generator, default_rng

from moptipy.examples.bitstrings.twomax import TwoMax
from moptipy.spaces.bitstrings import BitStrings
from moptipy.tests.objective import validate_objective
from moptipy.tests.on_bitstrings import random_bit_string


def test_twomax() -> None:
    """Test the twomax objective function."""
    random: Final[Generator] = default_rng()
    space: BitStrings = BitStrings(int(random.integers(2, 32)))
    f: TwoMax = TwoMax(space.dimension)

    validate_objective(
        objective=f,
        solution_space=space,
        make_solution_space_element_valid=random_bit_string,
        is_deterministic=True,
        lower_bound_threshold=0)
