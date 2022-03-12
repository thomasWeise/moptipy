"""Test random walks."""
# noinspection PyPackageRequirements
from moptipy.algorithms.random_walk import RandomWalk
from moptipy.examples.jssp.instance import Instance
from moptipy.operators.pwr.op0_shuffle import Op0Shuffle
from moptipy.operators.pwr.op1_swap2 import Op1Swap2
from moptipy.spaces.permutationswr import PermutationsWithRepetitions
from moptipy.tests.on_jssp import validate_algorithm_on_jssp


def test_random_walk_on_jssp():
    """Validate a random walk on the JSSP."""
    def create(instance: Instance,
               search_space: PermutationsWithRepetitions):
        assert isinstance(instance, Instance)
        assert isinstance(search_space, PermutationsWithRepetitions)
        return RandomWalk(Op0Shuffle(search_space), Op1Swap2())

    validate_algorithm_on_jssp(algorithm=create)
