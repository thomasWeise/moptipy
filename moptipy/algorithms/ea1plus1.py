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
        # create records for old and new point in the search space
        best_x = process.create()
        new_x = process.create()
        # obtain the random number generator
        random: Final[Generator] = process.get_random()

        # Resolving things such as "process." or "self." costs time.
        # We shovel a lot of function references into local variables
        # to save time.
        evaluate: Final[Callable] = process.evaluate
        op1: Final[Callable] = self.op1.op1
        should_terminate: Final[Callable] = process.should_terminate

        # Start at a random point in the search space and evaluate it.
        self.op0.op0(random, best_x)  # create one solution randomly
        best_f: Union[int, float] = evaluate(best_x)  # and evaluate it

        while not should_terminate():  # until we need to quit...
            op1(random, new_x, best_x)  # new_x = neighbor of best_x
            new_f: Union[int, float] = evaluate(new_x)
            if new_f <= best_f:  # new_x is no worse than best_x?
                best_f = new_f  # use its objective value
                best_x, new_x = new_x, best_x  # swap best and new

    def __str__(self) -> str:
        """
        Get the name of this (1+1)-EA.

        :return: "ea1p1" + any non-standard operator suffixes
        :rtype: str
        """
        return f"ea1p1{super().__str__()}"
