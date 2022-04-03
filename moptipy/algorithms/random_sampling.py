"""The random sampling algorithm."""
from typing import Final, Callable

from numpy.random import Generator

from moptipy.api.algorithm import Algorithm0
from moptipy.api.process import Process


class RandomSampling(Algorithm0):
    """In each step, random sampling creates a new, random solution."""

    def solve(self, process: Process) -> None:
        """
        Apply the random sampling approach to the given black-box process.

        :param process: the process object
        """
        x: Final = process.create()  # record for solution
        # obtain the random number generator
        random: Final[Generator] = process.get_random()

        # Resolving things such as "process." or "self." costs time.
        # We shovel a lot of function references into local variables
        # to save time.
        evaluate: Final[Callable] = process.evaluate
        op0: Final[Callable] = self.op0.op0
        should_terminate: Final[Callable] = process.should_terminate

        while not should_terminate():  # until we need to quit...
            op0(random, x)  # sample a random solution
            evaluate(x)  # evaluate its quality... but ignore this info

    def __str__(self) -> str:
        """
        Get the name of this random sampler.

        :return: "rs" + any non-standard operator suffixes
        """
        return f"rs{super().__str__()}"
