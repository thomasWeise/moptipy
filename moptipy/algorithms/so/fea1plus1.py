"""The (1+1)-FEA."""
from typing import Final, Callable, cast

import numpy as np
from numpy.random import Generator

from moptipy.api.algorithm import Algorithm1
from moptipy.api.operators import Op0, Op1
from moptipy.api.process import Process


class FEA1plus1(Algorithm1):
    """
    The FFA-based version of the (1+1)-EA: the (1+1)-FEA.

    This algorithm applies Frequency Fitness Assignment (FFA).
    This means that it does not select solutions based on whether
    they are better or worse. Instead, it selects the solution whose
    objective value has been encountered during the search less often.
    The word "best" therefore is not used in the traditional sense, i.e.,
    that one solution is better than another one terms of its objective
    value. Instead, the current best solution is always the one whose
    objective value we have seen the least often.

    In each step, a (1+1)-FEA creates a modified copy `new_x` of the
    current best solution `best_x`. It then increments the frequency fitness
    of both solutions by 1. If frequency fitness of `new_x` is not bigger
    the one of `best_x`, it becomes the new `best_x`.
    Otherwise, it is discarded.

    This algorithm implementation requires that objective values are
    integers and have lower and upper bounds that are not too far
    away from each other.
    """

    def __init__(self, op0: Op0, op1: Op1) -> None:
        """
        Create the (1+1)-FEA.

        :param op0: the nullary search operator
        :param op1: the unary search operator
        """
        super().__init__("fea1p1", op0, op1)

    def solve(self, process: Process) -> None:
        """
        Apply the (1+1)-FEA to an optimization problem.

        :param process: the black-box process object
        """
        # Create records for old and new point in the search space.
        best_x = process.create()  # record for best-so-far solution
        new_x = process.create()  # record for new solution
        lb: Final[int] = cast(int, process.lower_bound())

        # h holds the encounter frequency of each objective value.
        h: Final[np.ndarray] = np.zeros(
            cast(int, process.upper_bound()) - lb + 1, np.uint64)
        # Obtain the random number generator.
        random: Final[Generator] = process.get_random()

        # Put function references in variables to save time.
        evaluate: Final[Callable] = process.evaluate  # the objective
        op1: Final[Callable] = self.op1.op1  # the unary operator
        should_terminate: Final[Callable] = process.should_terminate

        # Start at a random point in the search space and evaluate it.
        self.op0.op0(random, best_x)  # Create 1 solution randomly and
        best_f: int = cast(int, evaluate(best_x)) - lb  # evaluate it.

        while not should_terminate():  # Until we need to quit...
            op1(random, new_x, best_x)  # new_x = neighbor of best_x
            new_f: int = cast(int, evaluate(new_x)) - lb

            h[new_f] = h[new_f] + 1  # Increase frequency of new_f and
            h[best_f] = best_h = h[best_f] + 1  # of best_f.
            if h[new_f] <= best_h:  # new_x is no worse than best_x?
                best_f = new_f  # Store its objective value.
                best_x, new_x = new_x, best_x  # Swap best and new.
