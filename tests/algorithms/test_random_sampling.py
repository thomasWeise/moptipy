"""Test random sampling."""
# noinspection PyPackageRequirements
from moptipy.algorithms.random_sampling import RandomSampling
from moptipy.examples.jssp.instance import JSSPInstance
from moptipy.operators.pwr.op0_shuffle import Op0Shuffle
from moptipy.tests.algorithm import check_algorithm_on_jssp


def test_random_sampling_jssp():
    def create(instance: JSSPInstance):
        assert isinstance(instance, JSSPInstance)
        return RandomSampling(Op0Shuffle())

    check_algorithm_on_jssp(algorithm=create)
