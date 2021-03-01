# noinspection PyPackageRequirements
from moptipy.algorithms.random_walk import RandomWalk
from moptipy.examples.jssp.instance import JSSPInstance
from moptipy.operators.pwr.op0 import Op0
from moptipy.operators.pwr.op1_swap2 import Op1Swap2
from moptipy.tests.algorithm import check_algorithm_on_jssp


def test_random_walk_jssp():
    def create(instance: JSSPInstance):
        assert isinstance(instance, JSSPInstance)
        return RandomWalk(Op0(), Op1Swap2())

    check_algorithm_on_jssp(algorithm=create)
