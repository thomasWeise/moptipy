"""The (1+1)-FEA."""
from typing import Final, Callable, cast

import numpy as np
from numpy.random import Generator

from moptipy.api.algorithm import Algorithm1
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

    def solve(self, process: Process) -> None:
        """
        Apply the (1+1)-FEA to the given black-box process.

        :param moptipy.api.Process process: the process object
        """
        # create the records for the best and new point in the search space
        best_x: Final = process.create()
        new_x: Final = process.create()
        lb: Final[int] = cast(int, process.lower_bound())

        # h holds the encounter frequency of each objective value.
        # By picking 32-bit integers as frequencies, we can do up to
        # 4 billion FEs before the frequency fitness becomes unreliable.
        h: Final[np.ndarray] = np.zeros(
            cast(int, process.upper_bound()) - lb + 1, np.uint32)
        # obtain the random number generator
        random: Final[Generator] = process.get_random()

        # Resolving things such as "process." or "self." costs time.
        # We shovel a lot of function references into local variables
        # to save time.
        evaluate: Final[Callable] = process.evaluate
        op1: Final[Callable] = self.op1.op1
        copy: Final[Callable] = process.copy
        should_terminate: Final[Callable] = process.should_terminate

        # If the process already knows one solution, then we will copy
        # it and start from there. Otherwise, we will start a random point
        # in the search space.
        best_f: int  # the best objective value
        new_f: int  # the objective value of the new solution
        best_h: int  # the frequency fitness corresponding to best_f
        if process.has_current_best():  # there already is a solution
            process.get_copy_of_current_best_x(best_x)  # get a copy
            best_f = cast(int, process.get_current_best_f()) - lb
        else:  # nope, no solution already known
            self.op0.op0(random, best_x)  # create one randomly
            best_f = cast(int, evaluate(best_x)) - lb  # and evaluate it

        while not should_terminate():  # until we need to quit...
            op1(random, new_x, best_x)  # new_x = neighbor of best_x
            new_f = cast(int, evaluate(new_x)) - lb

            h[new_f] = h[new_f] + 1  # increase frequency of new_f
            h[best_f] = best_h = h[best_f] + 1  # increase frequency of best_f
            if h[new_f] <= best_h:  # new_x is no worse than best_x?
                best_f = new_f  # use its objective value
                copy(best_x, new_x)  # and copy it to best_x

    def __str__(self) -> str:
        """
        Get the name of this (1+1)-FEA.

        :return: "fea1p1" + any non-standard operator suffixes
        :rtype: str
        """
        return f"fea1p1{super().__str__()}"
