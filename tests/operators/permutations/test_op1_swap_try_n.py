"""Test the unary swap-try-n operation."""

from moptipy.operators.permutations.op1_swap_try_n import Op1SwapTryN
from moptipy.tests.on_permutations import (
    validate_op1_with_step_size_on_permutations,
)


def test_op1_swaptn() -> None:
    """Test the unary swap-exactly-n operation."""

    def _min_unique(samples, pwrx) -> int:
        return max(1, min(samples, pwrx.n()) // 2)

    validate_op1_with_step_size_on_permutations(
        Op1SwapTryN, None, _min_unique,
        [0.0, 0.1, 0.2, 0.5, 1.0],
        perm_filter=lambda p: 2 <= p.dimension <= 2000)
