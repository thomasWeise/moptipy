"""Test the NSGA-II."""
from numpy.random import Generator, default_rng

from moptipy.algorithms.mo.nsga2 import NSGA2
from moptipy.api.mo_problem import MOProblem
from moptipy.examples.jssp.instance import Instance
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.operators.bitstrings.op1_flip1 import Op1Flip1
from moptipy.operators.bitstrings.op2_uniform import Op2Uniform
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle
from moptipy.operators.permutations.op1_swap2 import Op1Swap2
from moptipy.operators.permutations.op2_ox2 import Op2OrderBased
from moptipy.spaces.bitstrings import BitStrings
from moptipy.spaces.permutations import Permutations
from moptipy.tests.on_bitstrings import (
    validate_mo_algorithm_on_2_bitstring_problems,
    validate_mo_algorithm_on_3_bitstring_problems,
)
from moptipy.tests.on_jssp import validate_mo_algorithm_on_jssp


def test_nsga2_on_bitstrings() -> None:
    """Validate the MO-RLS on bit strings problem."""

    def create(bs: BitStrings, problem: MOProblem) -> NSGA2:
        assert isinstance(bs, BitStrings)
        assert isinstance(problem, MOProblem)
        random: Generator = default_rng()
        return NSGA2(Op0Random(), Op1Flip1(),
                     Op2Uniform(), int(random.integers(3, 10)),
                     float(random.uniform(0.1, 0.9)))

    validate_mo_algorithm_on_2_bitstring_problems(create)
    validate_mo_algorithm_on_3_bitstring_problems(create)


def test_nsga2_on_jssp() -> None:
    """Validate the NSGA-2 on the JSSP."""

    def create(instance: Instance, search_space: Permutations,
               problem: MOProblem) -> NSGA2:
        assert isinstance(instance, Instance)
        assert isinstance(search_space, Permutations)
        assert isinstance(problem, MOProblem)
        random: Generator = default_rng()
        return NSGA2(Op0Shuffle(search_space), Op1Swap2(),
                     Op2OrderBased(search_space), int(random.integers(3, 10)),
                     float(random.uniform(0.1, 0.9)))

    validate_mo_algorithm_on_jssp(create)
