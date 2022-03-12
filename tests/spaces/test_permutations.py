"""Test the space of permutations."""
from moptipy.spaces.permutations import Permutations
from moptipy.tests.space import validate_space


def test_permutations():
    """Test the permutation space."""
    def _invalid(x):
        x[0] = 22
        return x

    validate_space(Permutations(12), make_element_invalid=_invalid)
    validate_space(Permutations(1), make_element_invalid=_invalid)
