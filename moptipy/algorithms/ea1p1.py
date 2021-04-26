"""The (1+1)-EA."""
from typing import Final, Union

from numpy.random import Generator

from moptipy.api.algorithm import Algorithm1
from moptipy.api.process import Process


class EA1p1(Algorithm1):
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
        best_x: Final = process.create()
        new_x: Final = process.create()
        random: Final[Generator] = process.get_random()

        best_f: Union[int, float]
        if process.has_current_best():
            process.get_copy_of_current_best_x(best_x)
            best_f = process.get_current_best_f()
        else:
            self.op0.op0(random, best_x)
            best_f = process.evaluate(best_x)

        while not process.should_terminate():
            self.op1.op1(random, best_x, new_x)
            new_f: Union[int, float] = process.evaluate(new_x)
            if new_f <= best_f:
                best_f = new_f
                process.copy(new_x, best_x)

    def get_name(self) -> str:
        """
        Get the name of this (1+1)-EA.

        :return: "ea1p1" + any non-standard operator suffixes
        :rtype: str
        """
        name: Final[str] = super().get_name()
        return f"ea1p1_{name}" if (len(name) > 0) else "ea1p1"
