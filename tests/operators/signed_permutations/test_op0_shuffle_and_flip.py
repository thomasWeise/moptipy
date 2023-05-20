"""Test the nullary shuffle  and flip operation."""
from moptipy.operators.signed_permutations.op0_shuffle_and_flip import (
    Op0ShuffleAndFlip,
)
from moptipy.tests.on_signed_permutations import (
    validate_op0_on_signed_permutations,
)


def test_op0_shuffle() -> None:
    """Test the nullary shuffle operation."""
    validate_op0_on_signed_permutations(Op0ShuffleAndFlip)
