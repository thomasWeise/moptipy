"""Test the space of permutations."""
from moptipy.spaces.permutations import Permutations
from moptipy.tests.space import validate_space


def test_permutations():
    """Test the permutation space."""
    validate_space(Permutations(12))
    validate_space(Permutations(1))
