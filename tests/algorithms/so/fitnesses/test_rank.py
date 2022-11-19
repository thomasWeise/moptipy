"""Test the rank fitness assignment strategy."""

from moptipy.algorithms.so.fitnesses.rank import Rank
from moptipy.tests.on_bitstrings import validate_fitness_on_bitstrings


def test_rank_on_bit_strings() -> None:
    """Test the rank assignment process on bit strings."""
    validate_fitness_on_bitstrings(fitness=Rank())
