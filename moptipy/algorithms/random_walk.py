"""A random walk allgorithm implmentation."""
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
        x = process.create()
        random = process.get_random()

        if process.has_current_best():
            process.get_copy_of_current_best_x(x)
        else:
            self.op0.op0(random, x)

        while not process.should_terminate():
            self.op1.op1(random, x, x)
            process.evaluate(x)

    def get_name(self) -> str:
        """
        Get the name of this random walk.

        :return: "rw" + any non-standard operator suffixes
        :rtype: str
        """
        name = super().get_name()
        return f"rw_{name}" if (len(name) > 0) else "rw"
