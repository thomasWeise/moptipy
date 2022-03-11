"""The (1+1)-EA."""
from typing import Final, Union, Callable

from numpy.random import Generator

from moptipy.api.algorithm import Algorithm1
from moptipy.api.process import Process


class EA1plus1(Algorithm1):
    """
    A (1+1)-EA is a simple local search accepting all non-worsening moves.

    In each step, a (1+1)-EA creates a modified copy `new_x` of the
    current best solution `best_x`. If `new_x` is not worse than `best_x`,
    it becomes the new `best_x`. Otherwise, it is discarded.
    """

    def solve(self, process: Process) -> None:
        """
        Apply the (1+1)-EA to the given black-box process.

        :param moptipy.api.Process process: the process object
        """
        # create the records for the best and new point in the search space
        best_x: Final = process.create()
        new_x: Final = process.create()
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
        best_f: Union[int, float]
        if process.has_current_best():  # there already is a solution
            process.get_copy_of_current_best_x(best_x)  # get a copy
            best_f = process.get_current_best_f()  # and its quality
        else:  # nope, no solution already known
            self.op0.op0(random, best_x)  # create one randomly
            best_f = evaluate(best_x)  # and evaluate it

        while not should_terminate():  # until we need to quit...
            op1(random, new_x, best_x)  # new_x = neighbor of best_x
            new_f: Union[int, float] = evaluate(new_x)
            if new_f <= best_f:  # new_x is no worse than best_x?
                best_f = new_f  # use its objective value
                copy(best_x, new_x)  # and copy it to best_x

    def __str__(self) -> str:
        """
        Get the name of this (1+1)-EA.

        :return: "ea1p1" + any non-standard operator suffixes
        :rtype: str
        """
        return f"ea1p1{super().__str__()}"
