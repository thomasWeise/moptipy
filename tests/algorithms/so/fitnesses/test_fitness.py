"""Test the basic fitness assignment process."""

from moptipy.algorithms.so.fitness import Fitness
from moptipy.tests.on_bitstrings import validate_fitness_on_bitstrings


def test_fitness_on_bit_strings():
    """Test the basic fitness assignment process on bit strings."""
    validate_fitness_on_bitstrings(fitness=Fitness())
