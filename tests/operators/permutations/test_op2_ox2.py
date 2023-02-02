"""Test the binary OX2 crossover operation."""
from math import isqrt

from moptipy.operators.permutations.op2_ox2 import Op2OrderBased
from moptipy.tests.on_permutations import validate_op2_on_permutations


def test_op2_ox2() -> None:
    """Test the binary OX2 crossover operation."""
    validate_op2_on_permutations(
        Op2OrderBased, None,
        lambda samples, space: max(1, min(2, samples // 2,
                                          isqrt(space.n_points()))),
        lambda p: 3 < p.dimension < 75)
