"""Test the restarted Multi-Objective RLS."""
from typing import Callable, Final

from numpy.random import Generator

from moptipy.algorithms.restarts import restarts
from moptipy.api.algorithm import Algorithm1
from moptipy.api.mo_algorithm import MOAlgorithm
from moptipy.api.mo_problem import MOProblem
from moptipy.api.mo_process import MOProcess
from moptipy.api.operators import Op0, Op1
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


class __MORLS(Algorithm1, MOAlgorithm):
    """The MO-RLS is a local search accepting all non-worsening moves."""

    def solve_mo(self, process: MOProcess) -> None:
        """
        Apply the MO-RLS to an optimization problem.

        :param process: the black-box process object
        """
        # Create records for old and new point in the search space.
        best_x = process.create()  # record for best-so-far solution
        best_f = process.f_create()  # the objective values
        new_x = process.create()  # record for new solution
        new_f = process.f_create()  # the objective values
        # Obtain the random number generator.
        random: Final[Generator] = process.get_random()

        # Put function references in variables to save time.
        evaluate: Final[Callable] = process.f_evaluate  # the objective
        op1: Final[Callable] = self.op1.op1  # the unary operator
        should_terminate: Final[Callable] = process.should_terminate
        domination: Final[Callable] = process.f_dominates

        # Start at a random point in the search space and evaluate it.
        self.op0.op0(random, best_x)  # Create 1 solution randomly and
        evaluate(best_x, best_f)  # evaluate it.

        for _ in range(5):
            if should_terminate():  # Until we need to quit...
                return
            op1(random, new_x, best_x)  # new_x = neighbor of best_x
            evaluate(new_x, new_f)
            if domination(new_f, best_f) <= 0:  # new is not worse than best?
                best_f, new_f = new_f, best_f  # swap objective values.
                best_x, new_x = new_x, best_x  # swap best and new.

        process.check_in(best_x, best_f)  # check-in final result

    def __init__(self, op0: Op0, op1: Op1) -> None:
        """
        Create the randomized local search (rls).

        :param op0: the nullary search operator
        :param op1: the unary search operator
        """
        Algorithm1.__init__(self, "morls", op0, op1)

    def initialize(self) -> None:
        """Initialize the algorithm."""
        Algorithm1.initialize(self)


def test_restarted_morls_on_bitstrings() -> None:
    """Validate the restarted MO-RLS on bit strings problems."""

    def create(bs: BitStrings, problem: MOProblem) -> MOAlgorithm:
        assert isinstance(bs, BitStrings)
        assert isinstance(problem, MOProblem)
        return restarts(__MORLS(Op0Random(), Op1MoverNflip(
            bs.dimension, 1, True)))

    validate_mo_algorithm_on_2_bitstring_problems(create)
    validate_mo_algorithm_on_3_bitstring_problems(create)


def test_restarted_morls_on_jssp() -> None:
    """Validate the restarted MO-RLS on the JSSP."""

    def create(instance: Instance, search_space: Permutations,
               problem: MOProblem) -> __MORLS:
        assert isinstance(instance, Instance)
        assert isinstance(search_space, Permutations)
        assert isinstance(problem, MOProblem)
        return restarts(__MORLS(Op0Shuffle(search_space), Op1Swap2()))

    validate_mo_algorithm_on_jssp(create)
