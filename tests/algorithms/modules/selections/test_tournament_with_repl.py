"""Test the tournament selection algorithm with replacement."""

from moptipy.algorithms.modules.selections.tournament_with_repl import (
    TournamentWithReplacement,
)
from moptipy.tests.selection import validate_selection


def test_2_tournament_with_replacement() -> None:
    """Test the tournament selection strategy."""
    validate_selection(TournamentWithReplacement(2), False)


def test_3_tournament_with_replacement() -> None:
    """Test the tournament selection strategy."""
    validate_selection(TournamentWithReplacement(3), False)


def test_11_tournament_with_replacement() -> None:
    """Test the tournament selection strategy."""
    validate_selection(TournamentWithReplacement(11), False)
