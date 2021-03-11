"""Test random walks."""
# noinspection PyPackageRequirements
from moptipy.algorithms.random_walk import RandomWalk
from moptipy.examples.jssp.instance import Instance
from moptipy.operators.pwr.op0_shuffle import Op0Shuffle
from moptipy.operators.pwr.op1_swap2 import Op1Swap2
from moptipy.tests.algorithm import check_algorithm_on_jssp


def test_random_walk_jssp():
    def create(instance: Instance):
        assert isinstance(instance, Instance)
        return RandomWalk(Op0Shuffle(), Op1Swap2())

    check_algorithm_on_jssp(algorithm=create)
