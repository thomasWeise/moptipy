"""A multi-objectiveversion of the Randomized Local Search algorithm `rls`."""
from typing import Final, Callable

from numpy.random import Generator

from moptipy.api.algorithm import Algorithm1
from moptipy.api.mo_algorithm import MOAlgorithm
from moptipy.api.mo_process import MOProcess
from moptipy.api.mo_utils import domination
from moptipy.api.operators import Op0, Op1


class MORLS(Algorithm1, MOAlgorithm):
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

        # Start at a random point in the search space and evaluate it.
        self.op0.op0(random, best_x)  # Create 1 solution randomly and
        evaluate(best_x, best_f)  # evaluate it.

        while not should_terminate():  # Until we need to quit...
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
