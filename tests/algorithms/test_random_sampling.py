"""Test random sampling."""
import moptipy.tests as tst
# noinspection PyPackageRequirements
from moptipy.algorithms.random_sampling import RandomSampling
from moptipy.examples.jssp.instance import Instance
from moptipy.operators.pwr.op0_shuffle import Op0Shuffle


def test_random_sampling_jssp():
    def create(instance: Instance):
        assert isinstance(instance, Instance)
        return RandomSampling(Op0Shuffle())

    tst.test_algorithm_on_jssp(algorithm=create)
