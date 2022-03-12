"""Test the hill climber."""
# noinspection PyPackageRequirements
from moptipy.algorithms.hill_climber import HillClimber
from moptipy.examples.jssp.instance import Instance
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.operators.bitstrings.op1_m_over_n_flip import Op1MoverNflip
from moptipy.operators.pwr.op0_shuffle import Op0Shuffle
from moptipy.operators.pwr.op1_swap2 import Op1Swap2
from moptipy.spaces.bitstrings import BitStrings
from moptipy.spaces.permutationswr import PermutationsWithRepetitions
from moptipy.tests.on_bitstrings import validate_algorithm_on_onemax, \
    validate_algorithm_on_leadingones
from moptipy.tests.on_jssp import validate_algorithm_on_jssp


def test_hill_climber_on_jssp():
    """Validate the hill climber on the JSSP."""

    def create(instance: Instance,
               search_space: PermutationsWithRepetitions):
        assert isinstance(instance, Instance)
        assert isinstance(search_space, PermutationsWithRepetitions)
        return HillClimber(Op0Shuffle(search_space), Op1Swap2())

    validate_algorithm_on_jssp(create)


def test_hill_climber_on_onemax():
    """Validate the hill climber on the onemax problmem."""

    def create(bs: BitStrings):
        assert isinstance(bs, BitStrings)
        return HillClimber(Op0Random(), Op1MoverNflip(bs.dimension, 1, True))

    validate_algorithm_on_onemax(create)


def test_hill_climber_on_leadingones():
    """Validate the hill climber on the leadingones problmem."""

    def create(bs: BitStrings):
        assert isinstance(bs, BitStrings)
        return HillClimber(Op0Random(), Op1MoverNflip(bs.dimension, 1, True))

    validate_algorithm_on_leadingones(create)
