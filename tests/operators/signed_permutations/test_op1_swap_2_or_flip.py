"""Test the unary swap-2-or-flip operation."""
from moptipy.operators.signed_permutations.op1_swap_2_or_flip import (
    Op1Swap2OrFlip,
)
from moptipy.tests.on_signed_permutations import (
    validate_op1_on_signed_permutations,
)


def test_op1_swap2() -> None:
    """Test the unary swap-2-or-flip operation."""
    def _min_unique(samples, pwr) -> int:
        return max(1, min(samples, pwr.n()) // 2)
    validate_op1_on_signed_permutations(Op1Swap2OrFlip(),
                                        min_unique_samples=_min_unique)
