"""Test the Greedy(2+1)EAmod."""
from moptipy.algorithms.so.greedy_2plus1_ea_mod import GreedyTwoPlusOneEAmod
from moptipy.api.objective import Objective
from moptipy.examples.jssp.instance import Instance
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.operators.bitstrings.op1_m_over_n_flip import Op1MoverNflip
from moptipy.operators.bitstrings.op2_uniform import Op2Uniform
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle
from moptipy.operators.permutations.op1_swap2 import Op1Swap2
from moptipy.operators.permutations.op2_ox2 import Op2OrderBased
from moptipy.spaces.bitstrings import BitStrings
from moptipy.spaces.permutations import Permutations
from moptipy.tests.on_bitstrings import (
    validate_algorithm_on_leadingones,
    validate_algorithm_on_onemax,
)
from moptipy.tests.on_jssp import validate_algorithm_on_jssp


def test_greedy_2plus1_ea_mod_on_jssp() -> None:
    """Validate the Greedy(2+1)EAmod on the JSSP."""

    def create(instance: Instance, search_space: Permutations,
               objective: Objective) -> GreedyTwoPlusOneEAmod:
        assert isinstance(instance, Instance)
        assert isinstance(search_space, Permutations)
        assert isinstance(objective, Objective)
        return GreedyTwoPlusOneEAmod(Op0Shuffle(search_space), Op1Swap2(),
                                     Op2OrderBased(search_space))

    validate_algorithm_on_jssp(create)


def test_greedy_2plus1_ea_mod_on_onemax() -> None:
    """Validate the Greedy(2+1)EAmod on the OneMax problem."""

    def create(bs: BitStrings, objective: Objective) \
            -> GreedyTwoPlusOneEAmod:
        assert isinstance(bs, BitStrings)
        assert isinstance(objective, Objective)
        return GreedyTwoPlusOneEAmod(Op0Random(),
                                     Op1MoverNflip(bs.dimension, 1, True),
                                     Op2Uniform())

    validate_algorithm_on_onemax(create)


def test_greedy_2plus1_ea_mod_on_leadingones() -> None:
    """Validate the hill climber on the LeadingOnes problem."""

    def create(bs: BitStrings, objective: Objective) \
            -> GreedyTwoPlusOneEAmod:
        assert isinstance(bs, BitStrings)
        assert isinstance(objective, Objective)
        return GreedyTwoPlusOneEAmod(Op0Random(),
                                     Op1MoverNflip(bs.dimension, 1, True),
                                     Op2Uniform())

    validate_algorithm_on_leadingones(create)
