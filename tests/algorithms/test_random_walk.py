"""Test random walks."""
import moptipy.tests as tst
# noinspection PyPackageRequirements
from moptipy.algorithms.random_walk import RandomWalk
from moptipy.examples.jssp.instance import Instance
from moptipy.operators.pwr.op0_shuffle import Op0Shuffle
from moptipy.operators.pwr.op1_swap2 import Op1Swap2
from moptipy.spaces.permutationswr import PermutationsWithRepetitions


def test_random_walk_jssp():
    def create(instance: Instance,
               search_space: PermutationsWithRepetitions):
        assert isinstance(instance, Instance)
        assert isinstance(search_space, PermutationsWithRepetitions)
        return RandomWalk(Op0Shuffle(search_space), Op1Swap2())

    tst.test_algorithm_on_jssp(algorithm=create)
