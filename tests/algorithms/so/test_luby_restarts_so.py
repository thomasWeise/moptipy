"""Test the luby-restarted RLS."""


from moptipy.algorithms.luby_restarts import luby_restarts
from moptipy.algorithms.so.rls import RLS
from moptipy.api.algorithm import Algorithm
from moptipy.api.objective import Objective
from moptipy.examples.jssp.instance import Instance
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.operators.bitstrings.op1_m_over_n_flip import Op1MoverNflip
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle
from moptipy.operators.permutations.op1_swap2 import Op1Swap2
from moptipy.spaces.bitstrings import BitStrings
from moptipy.spaces.permutations import Permutations
from moptipy.tests.on_bitstrings import (
    validate_algorithm_on_onemax,
)
from moptipy.tests.on_jssp import validate_algorithm_on_jssp


def test_luby_restarted_rls_on_jssp() -> None:
    """Validate the luby-restarted RLS on the JSSP."""

    def create(instance: Instance, search_space: Permutations,
               objective: Objective) -> Algorithm:
        assert isinstance(instance, Instance)
        assert isinstance(search_space, Permutations)
        assert isinstance(objective, Objective)
        return luby_restarts(RLS(Op0Shuffle(search_space), Op1Swap2()), 5)

    validate_algorithm_on_jssp(create)


def test_luby_restarted_rls_on_onemax() -> None:
    """Validate the luby-restarted RLS on the OneMax problem."""

    def create(bs: BitStrings, objective: Objective) -> Algorithm:
        assert isinstance(bs, BitStrings)
        assert isinstance(objective, Objective)
        return luby_restarts(RLS(
            Op0Random(), Op1MoverNflip(bs.dimension, 1, True)), 5)

    validate_algorithm_on_onemax(create)
