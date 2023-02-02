"""Test the unary swap-n operation."""
from moptipy.operators.permutations.op1_swapn import Op1SwapN
from moptipy.tests.on_permutations import (
    validate_op1_on_permutations,
)


def test_op1_swapn() -> None:
    """Test the unary swap-n operation."""
    def _min_unique(samples, pwrx) -> int:
        return max(1, min(samples, pwrx.n()) // 2)
    validate_op1_on_permutations(Op1SwapN(), None, _min_unique,
                                 lambda p: len(set(p.blueprint)) > 2)
