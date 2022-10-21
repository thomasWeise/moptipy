"""Test the "best" selection algorithm."""

from moptipy.algorithms.modules.selections.best import Best
from moptipy.tests.selection import validate_selection


def test_truncation():
    """Test the best selection strategy."""
    validate_selection(Best(), True, True)
