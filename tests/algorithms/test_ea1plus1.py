"""Test the (1+1) EA)."""
# noinspection PyPackageRequirements
from moptipy.algorithms.ea1plus1 import EA1plus1
from moptipy.examples.jssp.instance import Instance
from moptipy.operators.pwr.op0_shuffle import Op0Shuffle
from moptipy.operators.pwr.op1_swap2 import Op1Swap2
from moptipy.spaces.permutationswr import PermutationsWithRepetitions
from moptipy.tests.algorithm import validate_algorithm_on_jssp


def test_ea1plus1_on_jssp():
    """Validate the (1+1)-EA on the JSSP."""
    def create(instance: Instance,
               search_space: PermutationsWithRepetitions):
        assert isinstance(instance, Instance)
        assert isinstance(search_space, PermutationsWithRepetitions)
        return EA1plus1(Op0Shuffle(search_space), Op1Swap2())

    validate_algorithm_on_jssp(create)
