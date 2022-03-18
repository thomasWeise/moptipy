"""Test the unary swap-2 operation."""

from moptipy.operators.permutations.op1_swap2 import Op1Swap2
from moptipy.tests.on_pwr import validate_op1_on_pwr


def test_op1_swap2():
    """Test the unary swap-2 operation."""
    def _min_unique(samples, pwr) -> int:
        return max(1, min(samples, pwr.n()) // 2)
    validate_op1_on_pwr(Op1Swap2(), min_unique_samples=_min_unique)
