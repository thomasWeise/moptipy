"""A random walk allgorithm implmentation."""
from typing import Final, Callable

from numpy.random import Generator

from moptipy.api.algorithm import Algorithm1
from moptipy.api.process import Process


class RandomWalk(Algorithm1):
    """
    Perform a random walk through the search space.

    In each step, a random walk creates a modified copy of the current
    solution and accepts it as starting point for the next step.
    """

    def solve(self, process: Process) -> None:
        """
        Apply the random walk to the given black-box process.

        :param moptipy.api.Process process: the process object
        """
        # create the records for the old and new point in the search space
        old_x = process.create()
        new_x = process.create()
        # obtain the random number generator
        random: Final[Generator] = process.get_random()

        # Resolving things such as "process." or "self." costs time.
        # We shovel a lot of function references into local variables
        # to save time.
        evaluate: Final[Callable] = process.evaluate
        op1: Final[Callable] = self.op1.op1
        should_terminate: Final[Callable] = process.should_terminate

        # If the process already knows one solution, then we will copy
        # it and start from there. Otherwise, we will start a random point
        # in the search space.
        if process.has_current_best():  # there already is a solution
            process.get_copy_of_current_best_x(new_x)  # get a copy
        else:  # nope, no solution already known
            self.op0.op0(random, new_x)  # create one randomly

        while not should_terminate():  # until we need to quit...
            old_x, new_x = new_x, old_x  # swap old and new solution
            op1(random, new_x, old_x)  # new_x = neighbor of old_x
            evaluate(new_x)

    def get_name(self) -> str:
        """
        Get the name of this random walk.

        :return: "rw" + any non-standard operator suffixes
        :rtype: str
        """
        return f"rw{super().get_name()}"
