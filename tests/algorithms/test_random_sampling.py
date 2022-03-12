"""Test random sampling."""
# noinspection PyPackageRequirements
from moptipy.algorithms.random_sampling import RandomSampling
from moptipy.examples.jssp.instance import Instance
from moptipy.operators.pwr.op0_shuffle import Op0Shuffle
from moptipy.spaces.permutationswr import PermutationsWithRepetitions
from moptipy.tests.on_jssp import validate_algorithm_on_jssp


def test_random_sampling_on_jssp():
    """Validate random sampling on the JSSP."""
    def create(instance: Instance,
               search_space: PermutationsWithRepetitions):
        assert isinstance(instance, Instance)
        assert isinstance(search_space, PermutationsWithRepetitions)
        return RandomSampling(Op0Shuffle(search_space))

    validate_algorithm_on_jssp(algorithm=create)
