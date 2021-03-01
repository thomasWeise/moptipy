# noinspection PyPackageRequirements
from moptipy.algorithms.random_sampling import RandomSampling
from moptipy.examples.jssp.instance import JSSPInstance
from moptipy.operators.pwr.op0 import Op0
from moptipy.tests.algorithm import check_algorithm_on_jssp


def test_random_sampling_jssp():
    def create(instance: JSSPInstance):
        assert isinstance(instance, JSSPInstance)
        return RandomSampling(Op0())

    check_algorithm_on_jssp(algorithm=create)
