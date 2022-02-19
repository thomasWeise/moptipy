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

        :param moptipy.api.Process process: the process object
        """
        # create the records for the best and new point in the search space
        best_x: Final = process.create()
        new_x: Final = process.create()
        # obtain the random number generator
        random: Final[Generator] = process.get_random()

        # Resolving things such as "process." or "self." costs time.
        # We shovel a lot of function references into local variables to save
        # time.
        evaluate: Final[Callable] = process.evaluate
        op1: Final[Callable] = self.op1.op1
        copy: Final[Callable] = process.copy
        should_terminate: Final[Callable] = process.should_terminate

        # Resolving things such as "process." or "self." costs time.
        # We shovel a lot of function references into local variables
        # to save time.
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
            if new_f < best_f:  # new_x is _better_ than best_x?
                best_f = new_f  # use its objective value
                copy(best_x, new_x)  # and copy it to best_x

    def get_name(self) -> str:
        """
        Get the name of this hill climber.

        :return: "hc" + any non-standard operator suffixes
        :rtype: str
        """
        return f"hc{super().get_name()}"
