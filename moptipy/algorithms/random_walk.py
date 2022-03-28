"""
A random walk algorithm implementation.

A random walk starts with a random point in the search space created with
the nullary search operator. In each step, it applies the unary search
operator to move a new point. It does not really care whether the new point
is better or worse, it will always accept it.

Of course, it still needs to call the objective function to make sure to
inform the :class:`Process` about the new point so that, at the end, we can
obtain the best point that was visited.
But during the course of its run, it will walk around the search space
randomly without direction.
"""
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
        # create records for old and new point in the search space
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

        # Start at a random point in the search space and evaluate it.
        self.op0.op0(random, new_x)  # create one solution randomly
        evaluate(new_x)  # and evaluate it

        while not should_terminate():  # until we need to quit...
            old_x, new_x = new_x, old_x  # swap old and new solution
            op1(random, new_x, old_x)  # new_x = neighbor of old_x
            evaluate(new_x)  # evaluate the solution, ignore result

    def __str__(self) -> str:
        """
        Get the name of this random walk.

        :return: "rw" + any non-standard operator suffixes
        :rtype: str
        """
        return f"rw{super().__str__()}"
