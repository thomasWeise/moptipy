"""Test the space of permutations."""
from moptipy.spaces.permutations import Permutations
from moptipy.tests.space import validate_space


def test_permutations_with_repetitions():
    """Test the permutations with repetitions."""
    def _invalid(x):
        x[0] = 33
        return x

    validate_space(Permutations.with_repetitions(10, 1),
                   make_element_invalid=_invalid)
    validate_space(Permutations.with_repetitions(2, 1),
                   make_element_invalid=_invalid)
    validate_space(Permutations.with_repetitions(2, 12),
                   make_element_invalid=_invalid)
    validate_space(Permutations.with_repetitions(21, 11),
                   make_element_invalid=_invalid)


def test_permutations():
    """Test the permutation space."""
    def _invalid(x):
        x[0] = 22
        return x

    validate_space(Permutations.standard(12), make_element_invalid=_invalid)
    validate_space(Permutations.standard(2), make_element_invalid=_invalid)
