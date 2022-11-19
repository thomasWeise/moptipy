"""Test the unary swap-2 operation."""
from moptipy.operators.permutations.op1_swap2 import Op1Swap2
from moptipy.tests.on_permutations import validate_op1_on_permutations


def test_op1_swap2() -> None:
    """Test the unary swap-2 operation."""
    def _min_unique(samples, pwr) -> int:
        return max(1, min(samples, pwr.n()) // 2)
    validate_op1_on_permutations(Op1Swap2(), min_unique_samples=_min_unique)
