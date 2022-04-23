"""Create one single random solution."""
from typing import Final

from moptipy.api.algorithm import Algorithm0
from moptipy.api.process import Process


# start book
class SingleRandomSample(Algorithm0):
    """
    This algorithm creates one single random solution.

    The single random sample algorithm applies the nullary search
    operator to sample exactly one single random solution.
    It then evaluates the solution by passing it to
    :meth:`~moptipy.api.process.Process.evaluate`. It does nothing
    else.

    This is a very very bad optimization algorithm. We only use it in
    our book to illustrate one basic concept for solving optimization
    problems: The generation of random solutions. The single-random
    sampling algorithm here is actually very wasteful: since it only
    generates exactly one single solution, it does not use its
    computational budget well. Even if you grant it 10'000 years, it
    will still only generate one solution. Even if it could generate
    and test thousands or millions of solutions, it will not do it.
    Nevertheless, after applying this "algorithm," you will have one
    valid solution remembered in the optimization process (embodied as
    instance `process` of :class:`~moptipy.api.process.Process`).

    This concept of random sampling is then refined in the
    :class:`~moptipy.algorithms.random_sampling.RandomSampling`
    algorithm, which repeats generating random solutions until its
    allotted runtime is exhausted.
    """

    def solve(self, process: Process) -> None:
        """
        Apply the single random sampling approach.

        :param process: the process object
        """
        x: Final = process.create()  # Create the solution record.
        self.op0.op0(process.get_random(), x)  # Create random solution
        process.evaluate(x)  # Evaluate that random solution.
# end book

    def __str__(self) -> str:
        """
        Get the name of this single random sampler.

        :return: "1rs" + any non-standard operator suffixes
        """
        return f"1rs{super().__str__()}"
