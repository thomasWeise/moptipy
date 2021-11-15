"""Test random sampling."""
import moptipy.tests as tst
# noinspection PyPackageRequirements
from moptipy.algorithms.random_sampling import RandomSampling
from moptipy.examples.jssp.instance import Instance
from moptipy.operators.pwr.op0_shuffle import Op0Shuffle
from moptipy.spaces.permutationswr import PermutationsWithRepetitions


def test_random_sampling_jssp():
    def create(instance: Instance,
               search_space: PermutationsWithRepetitions):
        assert isinstance(instance, Instance)
        assert isinstance(search_space, PermutationsWithRepetitions)
        return RandomSampling(Op0Shuffle(search_space))

    tst.test_algorithm_on_jssp(algorithm=create)
