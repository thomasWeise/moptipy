"""The random sampling algorithm."""
from moptipy.api.algorithm import Algorithm0
from moptipy.api.process import Process


class RandomSampling(Algorithm0):
    """In each step, random sampling creates a new, random solution."""

    def solve(self, process: Process) -> None:
        """
        Apply the random sampling approach to the given black-box process.

        :param moptipy.api.Process process: the process object
        """
        x = process.create()
        random = process.get_random()

        while not process.should_terminate():
            self.op0.op0(random, x)
            process.evaluate(x)

    def get_name(self) -> str:
        """
        Get the name of this random sampler.

        :return: "rs" + any non-standard operator suffixes
        :rtype: str
        """
        name = super().get_name()
        return ("rs_" + name) if (len(name) > 0) else "rs"
