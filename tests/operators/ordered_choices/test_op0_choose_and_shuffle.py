"""Test the nullary choose-and-shuffle operation."""
from moptipy.operators.ordered_choices.op0_choose_and_shuffle import (
    Op0ChooseAndShuffle,
)
from moptipy.tests.on_ordered_choices import validate_op0_on_choices


def test_op0_shuffle() -> None:
    """Test the nullary choose-and-shuffle operation."""
    validate_op0_on_choices(Op0ChooseAndShuffle)
