"""The hill climbing algorithm implementation."""
from typing import Final, Union, Callable

from numpy.random import Generator

from moptipy.api.algorithm import Algorithm1
from moptipy.api.process import Process


class HillClimber(Algorithm1):
    """
    The stochastic hill climbing algorithm only accepts improving moves.

    In each step, a hill climber creates a modified copy `new_x` of the
    current best solution `best_x`. If `new_x` is better than `best_x`,
    it becomes the new `best_x`. Otherwise, it is discarded.
    """

    def solve(self, process: Process) -> None:
        """
        Apply the hill climber to the given black-box process.

        :param process: the process object
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
            if new_f < best_f:  # new_x is _better_ than best_x?
                best_f = new_f  # use its objective value
                best_x, new_x = new_x, best_x  # swap best and new

    def __str__(self) -> str:
        """
        Get the name of this hill climber.

        :return: "hc" + any non-standard operator suffixes
        """
        return f"hc{super().__str__()}"
