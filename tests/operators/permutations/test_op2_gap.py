"""Test the binary GAP crossover operation."""
from math import isqrt

from moptipy.operators.permutations.op2_gap import (
    Op2GeneralizedAlternatingPosition,
)
from moptipy.tests.on_permutations import validate_op2_on_permutations


def test_op2_gap() -> None:
    """Test the binary GAP crossover operation."""
    validate_op2_on_permutations(
        Op2GeneralizedAlternatingPosition, None,
        lambda samples, space: max(1, min(2, samples // 2,
                                          isqrt(space.n_points()))),
        lambda p: p.dimension < 75)
