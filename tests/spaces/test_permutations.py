from moptipy.spaces import Permutations

from moptipy.tests.space import check_space


def test_permutations():
    check_space(Permutations(12))
    check_space(Permutations(1))
