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
        x: Final = process.create()
        self.op0.op0(process.get_random(), x)
        process.evaluate(x)

    def get_name(self) -> str:
        """
        Get the name of this single random sampler.

        :return: "1rs" + any non-standard operator suffixes
        :rtype: str
        """
        name: Final[str] = super().get_name()
        return f"1rs_{name}" if (len(name) > 0) else "1rs"
