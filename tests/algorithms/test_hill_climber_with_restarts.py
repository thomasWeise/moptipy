"""Test the hill climber with restarts."""
# noinspection PyPackageRequirements
from moptipy.algorithms.hill_climber_with_restarts import \
    HillClimberWithRestarts
from moptipy.examples.jssp.instance import Instance
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.operators.bitstrings.op1_m_over_n_flip import Op1MoverNflip
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle
from moptipy.operators.permutations.op1_swap2 import Op1Swap2
from moptipy.spaces.bitstrings import BitStrings
from moptipy.spaces.permutations import Permutations
from moptipy.tests.on_bitstrings import validate_algorithm_on_onemax, \
    validate_algorithm_on_leadingones
from moptipy.tests.on_jssp import validate_algorithm_on_jssp


def test_hill_climber_with_restarts_on_jssp():
    """Validate the hill climber with restarts on the JSSP."""

    def create(instance: Instance,
               search_space: Permutations):
        assert isinstance(instance, Instance)
        assert isinstance(search_space, Permutations)
        return HillClimberWithRestarts(Op0Shuffle(search_space), Op1Swap2(),
                                       200)

    validate_algorithm_on_jssp(create)


def test_hill_climber_with_restarts_on_onemax():
    """Validate the hill climber with restarts on the OneMax Problem."""

    def create(bs: BitStrings):
        assert isinstance(bs, BitStrings)
        return HillClimberWithRestarts(
            Op0Random(), Op1MoverNflip(bs.dimension, 1, True), 500)

    validate_algorithm_on_onemax(create)


def test_hill_climber_with_restarts_on_leadingones():
    """Validate the hill climber with restarts on the LeadingOnes problem."""

    def create(bs: BitStrings):
        assert isinstance(bs, BitStrings)
        return HillClimberWithRestarts(
            Op0Random(), Op1MoverNflip(bs.dimension, 1, True), 33)

    validate_algorithm_on_leadingones(create)
