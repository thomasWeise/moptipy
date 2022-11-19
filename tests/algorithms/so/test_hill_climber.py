"""Test the hill climber."""
from moptipy.algorithms.so.hill_climber import HillClimber
from moptipy.api.objective import Objective
from moptipy.examples.jssp.instance import Instance
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.operators.bitstrings.op1_m_over_n_flip import Op1MoverNflip
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle
from moptipy.operators.permutations.op1_swap2 import Op1Swap2
from moptipy.spaces.bitstrings import BitStrings
from moptipy.spaces.permutations import Permutations
from moptipy.tests.on_bitstrings import (
    validate_algorithm_on_leadingones,
    validate_algorithm_on_onemax,
)
from moptipy.tests.on_jssp import validate_algorithm_on_jssp


def test_hill_climber_on_jssp() -> None:
    """Validate the hill climber on the JSSP."""

    def create(instance: Instance, search_space: Permutations,
               objective: Objective) -> HillClimber:
        assert isinstance(instance, Instance)
        assert isinstance(search_space, Permutations)
        assert isinstance(objective, Objective)
        return HillClimber(Op0Shuffle(search_space), Op1Swap2())

    validate_algorithm_on_jssp(create)


def test_hill_climber_on_onemax() -> None:
    """Validate the hill climber on the OneMax problem."""

    def create(bs: BitStrings, objective: Objective) -> HillClimber:
        assert isinstance(bs, BitStrings)
        assert isinstance(objective, Objective)
        return HillClimber(Op0Random(), Op1MoverNflip(bs.dimension, 1, True))

    validate_algorithm_on_onemax(create)


def test_hill_climber_on_leadingones() -> None:
    """Validate the hill climber on the LeadingOnes problem."""

    def create(bs: BitStrings, objective: Objective) -> HillClimber:
        assert isinstance(bs, BitStrings)
        assert isinstance(objective, Objective)
        return HillClimber(Op0Random(), Op1MoverNflip(bs.dimension, 1, True))

    validate_algorithm_on_leadingones(create)
