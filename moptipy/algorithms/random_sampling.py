from moptipy.api.process import Process
from moptipy.api.algorithm import Algorithm0


class RandomSampling(Algorithm0):
    """
    In each step, a random sampling algorithm creates a new, random solution.
    """

    def solve(self, process: Process) -> None:
        x = process.create()
        random = process.get_random()

        while not process.should_terminate():
            self.op0.op0(random, x)
            process.evaluate(x)

    def get_name(self) -> str:
        name = super().get_name()
        return ("rs_" + name) if (len(name) > 0) else "rs"
