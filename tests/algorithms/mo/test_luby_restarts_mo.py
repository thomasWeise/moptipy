"""Test the Multi-Objective RLS."""
from moptipy.algorithms.luby_restarts import luby_restarts
from moptipy.algorithms.mo.morls import MORLS
from moptipy.api.mo_algorithm import MOAlgorithm
from moptipy.api.mo_problem import MOProblem
from moptipy.examples.jssp.instance import Instance
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.operators.bitstrings.op1_m_over_n_flip import Op1MoverNflip
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle
from moptipy.operators.permutations.op1_swap2 import Op1Swap2
from moptipy.spaces.bitstrings import BitStrings
from moptipy.spaces.permutations import Permutations
from moptipy.tests.on_bitstrings import (
    validate_mo_algorithm_on_2_bitstring_problems,
    validate_mo_algorithm_on_3_bitstring_problems,
)
from moptipy.tests.on_jssp import validate_mo_algorithm_on_jssp


def test_luby_restarted_morls_on_bitstrings() -> None:
    """Validate the luby-restarted MO-RLS on bit strings problems."""

    def create(bs: BitStrings, problem: MOProblem) -> MOAlgorithm:
        assert isinstance(bs, BitStrings)
        assert isinstance(problem, MOProblem)
        return luby_restarts(
            MORLS(Op0Random(), Op1MoverNflip(bs.dimension, 1, True)), 5)

    validate_mo_algorithm_on_2_bitstring_problems(create)
    validate_mo_algorithm_on_3_bitstring_problems(create)


def test_luby_restarted_morls_on_jssp() -> None:
    """Validate the luby-restarted MO-RLS on the JSSP."""

    def create(instance: Instance, search_space: Permutations,
               problem: MOProblem) -> MOAlgorithm:
        assert isinstance(instance, Instance)
        assert isinstance(search_space, Permutations)
        assert isinstance(problem, MOProblem)
        return luby_restarts(MORLS(Op0Shuffle(search_space), Op1Swap2()), 5)

    validate_mo_algorithm_on_jssp(create)
