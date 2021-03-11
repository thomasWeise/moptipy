"""Test random sampling."""
# noinspection PyPackageRequirements
from moptipy.algorithms.random_sampling import RandomSampling
from moptipy.examples.jssp.instance import Instance
from moptipy.operators.pwr.op0_shuffle import Op0Shuffle
from moptipy.tests.algorithm import check_algorithm_on_jssp


def test_random_sampling_jssp():
    def create(instance: Instance):
        assert isinstance(instance, Instance)
        return RandomSampling(Op0Shuffle())

    check_algorithm_on_jssp(algorithm=create)
