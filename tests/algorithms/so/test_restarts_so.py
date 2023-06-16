"""Test the restarted RLS."""
from typing import Callable, Final

from numpy.random import Generator

from moptipy.algorithms.restarts import restarts
from moptipy.api.algorithm import Algorithm, Algorithm1
from moptipy.api.objective import Objective
from moptipy.api.operators import Op0, Op1
from moptipy.api.process import Process
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


class __RLS(Algorithm1):
    """A time-limited RLS algorithm."""

    def solve(self, process: Process) -> None:
        """
        Apply the RLS to an optimization problem.

        :param process: the black-box process object
        """
        # Create records for old and new point in the search space.
        best_x = process.create()  # record for best-so-far solution
        new_x = process.create()  # record for new solution
        # Obtain the random number generator.
        random: Final[Generator] = process.get_random()

        # Put function references in variables to save time.
        evaluate: Final[Callable] = process.evaluate  # the objective
        op1: Final[Callable] = self.op1.op1  # the unary operator
        should_terminate: Final[Callable] = process.should_terminate

        # Start at a random point in the search space and evaluate it.
        self.op0.op0(random, best_x)  # Create 1 solution randomly and
        best_f: int | float = evaluate(best_x)  # evaluate it.

        for _i in range(5):
            if should_terminate():  # Until we need to quit...
                return
            op1(random, new_x, best_x)  # new_x = neighbor of best_x
            new_f: int | float = evaluate(new_x)
            if new_f <= best_f:  # new_x is not worse than best_x?
                best_f = new_f  # Store its objective value.
                best_x, new_x = new_x, best_x  # Swap best and new.

    def __init__(self, op0: Op0, op1: Op1) -> None:
        """
        Create the randomized local search (rls).

        :param op0: the nullary search operator
        :param op1: the unary search operator
        """
        super().__init__("rls", op0, op1)


def test_restarted_rls_on_jssp() -> None:
    """Validate the restarted RLS on the JSSP."""

    def create(instance: Instance, search_space: Permutations,
               objective: Objective) -> Algorithm:
        assert isinstance(instance, Instance)
        assert isinstance(search_space, Permutations)
        assert isinstance(objective, Objective)
        return restarts(__RLS(Op0Shuffle(search_space), Op1Swap2()))

    validate_algorithm_on_jssp(create)


def test_restarted_rls_on_onemax() -> None:
    """Validate the restarted RLS on the OneMax problem."""

    def create(bs: BitStrings, objective: Objective) -> Algorithm:
        assert isinstance(bs, BitStrings)
        assert isinstance(objective, Objective)
        return restarts(__RLS(
            Op0Random(), Op1MoverNflip(bs.dimension, 1, True)))

    validate_algorithm_on_onemax(create)
