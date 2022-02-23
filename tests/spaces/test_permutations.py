"""Test the space of permutations."""
import moptipy.tests.space as tst
from moptipy.spaces.permutations import Permutations


def test_permutations():
    """Test the permutation space."""
    tst.test_space(Permutations(12))
    tst.test_space(Permutations(1))
