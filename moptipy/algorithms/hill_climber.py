"""The hill climbing algorithm implementation."""
from typing import Final, Union

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
            if new_f < best_f:
                best_f = new_f
                process.copy(best_x, new_x)

    def get_name(self) -> str:
        """
        Get the name of this hill climber.

        :return: "hc" + any non-standard operator suffixes
        :rtype: str
        """
        return f"hc{super().get_name()}"
