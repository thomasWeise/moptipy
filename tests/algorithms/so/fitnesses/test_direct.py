"""Test the direct fitness assignment strategy."""

from moptipy.algorithms.so.fitnesses.direct import Direct
from moptipy.tests.on_bitstrings import validate_fitness_on_bitstrings


def test_direct_on_bit_strings() -> None:
    """Test the direct assignment process on bit strings."""
    validate_fitness_on_bitstrings(fitness=Direct())
