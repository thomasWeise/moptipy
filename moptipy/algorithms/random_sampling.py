"""The random sampling algorithm."""
from typing import Final

from numpy.random import Generator

from moptipy.api.algorithm import Algorithm0
from moptipy.api.process import Process


class RandomSampling(Algorithm0):
    """In each step, random sampling creates a new, random solution."""

    def solve(self, process: Process) -> None:
        """
        Apply the random sampling approach to the given black-box process.

        :param moptipy.api.Process process: the process object
        """
        x: Final = process.create()
        random: Final[Generator] = process.get_random()

        while not process.should_terminate():
            self.op0.op0(random, x)
            process.evaluate(x)

    def get_name(self) -> str:
        """
        Get the name of this random sampler.

        :return: "rs" + any non-standard operator suffixes
        :rtype: str
        """
        return f"rs{super().get_name()}"
