"""Test the binary order crossover operation."""
from math import isqrt

from moptipy.operators.permutations.op2_order import Op2Order
from moptipy.tests.on_permutations import permutations_for_tests, \
    validate_op2_on_1_permutations


def test_op2_order():
    """Test the binary order crossover operation."""
    cumm_size: int = 0
    for pwr in permutations_for_tests():
        if pwr.dimension < 4:
            continue  # too small, skip
        if pwr.dimension > 50:  # too big, takes too long
            if (pwr.dimension > 100) or (cumm_size > 200):
                continue
        cumm_size += pwr.dimension

        validate_op2_on_1_permutations(
            Op2Order, pwr, None,
            min_unique_samples=lambda samples, space:
            max(1, min(2, samples // 2, isqrt(space.n_points()))))
