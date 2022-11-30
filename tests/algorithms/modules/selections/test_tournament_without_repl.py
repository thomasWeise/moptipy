"""Test the tournament selection algorithm without replacement."""

from moptipy.algorithms.modules.selections.tournament_without_repl import (
    TournamentWithoutReplacement,
)
from moptipy.tests.selection import validate_selection


def test_2_tournament_with_replacement() -> None:
    """Test the tournament selection strategy."""
    validate_selection(TournamentWithoutReplacement(2), False)


def test_3_tournament_with_replacement() -> None:
    """Test the tournament selection strategy."""
    validate_selection(TournamentWithoutReplacement(3), False)


def test_11_tournament_with_replacement() -> None:
    """Test the tournament selection strategy."""
    validate_selection(TournamentWithoutReplacement(11), False)
