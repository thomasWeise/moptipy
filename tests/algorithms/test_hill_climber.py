# noinspection PyPackageRequirements
from moptipy.algorithms.hill_climber import HillClimber
from moptipy.examples.jssp.instance import JSSPInstance
from moptipy.operators.pwr.op0 import Op0
from moptipy.operators.pwr.op1_swap2 import Op1Swap2
from moptipy.tests.algorithm import check_algorithm_on_jssp


def test_hill_climber_jssp():
    def create(instance: JSSPInstance):
        assert isinstance(instance, JSSPInstance)
        return HillClimber(Op0(), Op1Swap2())

    check_algorithm_on_jssp(algorithm=create)
