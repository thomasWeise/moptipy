"""Test the unary insertion operation."""

# noinspection PyProtectedMember
from moptipy.operators.permutations.op1_insert1 import (
    Op1Insert1,
)
from moptipy.tests.on_permutations import validate_op1_on_permutations


def test_op1_insert1() -> None:
    """Test the unary insertion operation."""

    def _min_unique(samples, pwr) -> int:
        return max(1, min(samples, pwr.n()) // 2)

    validate_op1_on_permutations(Op1Insert1(), min_unique_samples=_min_unique)
