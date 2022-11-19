"""Test the basic fitness assignment process."""

from moptipy.algorithms.so.fitnesses.rank_and_iteration import RankAndIteration
from moptipy.tests.on_bitstrings import validate_fitness_on_bitstrings


def test_rank_and_iteration_on_bit_strings() -> None:
    """Test the basic fitness assignment process on bit strings."""
    validate_fitness_on_bitstrings(fitness=RankAndIteration())
