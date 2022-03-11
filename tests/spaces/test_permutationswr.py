"""Test the space of permutations with repetitions."""
from moptipy.spaces.permutationswr import PermutationsWithRepetitions
from moptipy.tests.space import validate_space


def test_permutations_with_repetitions():
    """Test the permutations with repetitions."""
    validate_space(PermutationsWithRepetitions(10, 1))
    validate_space(PermutationsWithRepetitions(2, 1))
    validate_space(PermutationsWithRepetitions(2, 12))
    validate_space(PermutationsWithRepetitions(21, 11))
