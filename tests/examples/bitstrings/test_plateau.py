"""Test the plateau objective function."""

from moptipy.examples.bitstrings.plateau import Plateau
from moptipy.spaces.bitstrings import BitStrings
from moptipy.tests.objective import validate_objective
from moptipy.tests.on_bitstrings import random_bit_string


def test_plateau() -> None:
    """Test the plateau objective function."""
    for n in range(3, 8):
        for k in range(2, n >> 1):
            f: Plateau = Plateau(n, k)
            space: BitStrings = f.space()

            validate_objective(
                objective=f,
                solution_space=space,
                make_solution_space_element_valid=random_bit_string,
                is_deterministic=True,
                lower_bound_threshold=0)
