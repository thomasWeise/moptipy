"""Test the nullary shuffle operation."""

from moptipy.operators.pwr.op0_shuffle import Op0Shuffle
from moptipy.tests.on_pwr import validate_op0_on_pwr


def test_op0_shuffle():
    """Test the nullary shuffle operation."""
    validate_op0_on_pwr(Op0Shuffle)
