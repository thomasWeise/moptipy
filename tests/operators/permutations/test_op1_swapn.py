"""Test the unary swap-n operation."""
from moptipy.operators.permutations.op1_swapn import Op1SwapN
from moptipy.tests.on_permutations import (
    permutations_for_tests,
    validate_op1_on_1_permutations,
)


def test_op1_swapn():
    """Test the unary swap-n operation."""
    def _min_unique(samples, pwrx) -> int:
        return max(1, min(samples, pwrx.n()) // 2)
    op1 = Op1SwapN()
    for pwr in permutations_for_tests():
        if len(set(pwr.blueprint)) <= 2:  # we need > 2 values
            continue
        validate_op1_on_1_permutations(op1, pwr, None, _min_unique)
