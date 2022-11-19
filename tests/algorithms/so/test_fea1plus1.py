"""Test the (1+1)-FEA."""
from moptipy.algorithms.so.fea1plus1 import FEA1plus1
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


def test_fea1plus1_on_jssp() -> None:
    """Validate the (1+1)-FEA on the JSSP."""

    def create(instance: Instance, search_space: Permutations,
               objective: Objective) -> FEA1plus1:
        assert isinstance(instance, Instance)
        assert isinstance(search_space, Permutations)
        assert isinstance(objective, Objective)
        return FEA1plus1(Op0Shuffle(search_space), Op1Swap2())

    validate_algorithm_on_jssp(create)


def test_fea1plus1_on_onemax() -> None:
    """Validate the (1+1)-FEA on the OneMax problem."""

    def create(bs: BitStrings, objective: Objective) -> FEA1plus1:
        assert isinstance(bs, BitStrings)
        assert isinstance(objective, Objective)
        return FEA1plus1(Op0Random(), Op1MoverNflip(bs.dimension, 1, True))

    validate_algorithm_on_onemax(create)


def test_fea1plus1_on_leadingones() -> None:
    """Validate the (1+1)-FEA on the LeadingOnes problem."""

    def create(bs: BitStrings, objective: Objective) -> FEA1plus1:
        assert isinstance(bs, BitStrings)
        assert isinstance(objective, Objective)
        return FEA1plus1(Op0Random(), Op1MoverNflip(bs.dimension, 1, True))

    validate_algorithm_on_leadingones(create)
