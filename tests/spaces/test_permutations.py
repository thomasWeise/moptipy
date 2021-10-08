"""Test the space of permutations."""
import moptipy.tests as tst
from moptipy.spaces import Permutations


def test_permutations():
    tst.test_space(Permutations(12))
    tst.test_space(Permutations(1))
