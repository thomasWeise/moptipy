"""Test the space of permutations with repetitions."""
from moptipy.spaces import PermutationsWithRepetitions

from moptipy.tests.space import check_space


def test_permutations():
    check_space(PermutationsWithRepetitions(10, 1))
    check_space(PermutationsWithRepetitions(1, 1))
    check_space(PermutationsWithRepetitions(1, 12))
    check_space(PermutationsWithRepetitions(21, 11))
