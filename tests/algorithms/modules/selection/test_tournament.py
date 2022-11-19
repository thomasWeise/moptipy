"""Test the tournament selection algorithm."""

from moptipy.algorithms.modules.selections.tournament import Tournament
from moptipy.tests.selection import validate_selection


def test_2_tournament_with_replacement() -> None:
    """Test the tournament selection strategy."""
    validate_selection(Tournament(2, True), False)


def test_2_tournament_without_replacement() -> None:
    """Test the tournament selection strategy."""
    validate_selection(Tournament(2, False), False, 2)


def test_3_tournament_with_replacement() -> None:
    """Test the tournament selection strategy."""
    validate_selection(Tournament(3, True), False)


def test_3_tournament_without_replacement() -> None:
    """Test the tournament selection strategy."""
    validate_selection(Tournament(3, False), False, 3)


def test_11_tournament_with_replacement() -> None:
    """Test the tournament selection strategy."""
    validate_selection(Tournament(11, True), False)


def test_11_tournament_without_replacement() -> None:
    """Test the tournament selection strategy."""
    validate_selection(Tournament(11, False), False, 11)
