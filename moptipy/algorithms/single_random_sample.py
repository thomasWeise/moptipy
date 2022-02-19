"""Create one single random solution."""
from typing import Final

from moptipy.api.algorithm import Algorithm0
from moptipy.api.process import Process


class SingleRandomSample(Algorithm0):
    """This algorithm creates one single random solution."""

    def solve(self, process: Process) -> None:
        """
        Apply the single random sampling approach.

        :param moptipy.api.Process process: the process object
        """
        x: Final = process.create()  # create the solution record
        self.op0.op0(process.get_random(), x)  # randomize contents
        process.evaluate(x)  # evaluate quality

    def get_name(self) -> str:
        """
        Get the name of this single random sampler.

        :return: "1rs" + any non-standard operator suffixes
        :rtype: str
        """
        return f"1rs{super().get_name()}"
