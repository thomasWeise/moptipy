"""Test the space of permutations with repetitions."""
from moptipy.spaces.permutationswr import PermutationsWithRepetitions
from moptipy.tests.space import validate_space


def test_permutations_with_repetitions():
    """Test the permutations with repetitions."""
    def _invalid(x):
        x[0] = 33
        return x

    validate_space(PermutationsWithRepetitions(10, 1),
                   make_element_invalid=_invalid)
    validate_space(PermutationsWithRepetitions(2, 1),
                   make_element_invalid=_invalid)
    validate_space(PermutationsWithRepetitions(2, 12),
                   make_element_invalid=_invalid)
    validate_space(PermutationsWithRepetitions(21, 11),
                   make_element_invalid=_invalid)
