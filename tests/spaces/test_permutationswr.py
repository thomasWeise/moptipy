"""Test the space of permutations with repetitions."""
import moptipy.tests as tst
from moptipy.spaces import PermutationsWithRepetitions


def test_permutations():
    tst.test_space(PermutationsWithRepetitions(10, 1))
    tst.test_space(PermutationsWithRepetitions(2, 1))
    tst.test_space(PermutationsWithRepetitions(2, 12))
    tst.test_space(PermutationsWithRepetitions(21, 11))
