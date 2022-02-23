"""Test the space of permutations with repetitions."""
import moptipy.tests.space as tst
from moptipy.spaces.permutationswr import PermutationsWithRepetitions


def test_permutations_with_repetitions():
    """Test the permutations with repetitions."""
    tst.test_space(PermutationsWithRepetitions(10, 1))
    tst.test_space(PermutationsWithRepetitions(2, 1))
    tst.test_space(PermutationsWithRepetitions(2, 12))
    tst.test_space(PermutationsWithRepetitions(21, 11))
