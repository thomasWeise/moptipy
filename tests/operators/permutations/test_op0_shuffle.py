"""Test the nullary shuffle operation."""
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle
from moptipy.tests.on_permutations import validate_op0_on_permutations


def test_op0_shuffle() -> None:
    """Test the nullary shuffle operation."""
    validate_op0_on_permutations(Op0Shuffle)
