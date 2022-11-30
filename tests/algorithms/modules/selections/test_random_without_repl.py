"""Test the random selection without replacement."""

from moptipy.algorithms.modules.selections.random_without_repl import (
    RandomWithoutReplacement,
)
from moptipy.tests.selection import validate_selection


def test_random_without_replacement() -> None:
    """Test the random selection without replacement strategy."""
    validate_selection(RandomWithoutReplacement(), True)
