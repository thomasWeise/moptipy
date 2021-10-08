"""Test the hill climber."""
import moptipy.tests as tst
# noinspection PyPackageRequirements
from moptipy.algorithms.hill_climber import HillClimber
from moptipy.examples.jssp.instance import Instance
from moptipy.operators.pwr.op0_shuffle import Op0Shuffle
from moptipy.operators.pwr.op1_swap2 import Op1Swap2


def test_hill_climber_jssp():
    def create(instance: Instance):
        assert isinstance(instance, Instance)
        return HillClimber(Op0Shuffle(), Op1Swap2())

    tst.test_algorithm_on_jssp(algorithm=create)
