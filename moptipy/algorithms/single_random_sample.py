"""
The `1rs` "algorithm" creates one single random solution.

The single random sample algorithm applies the nullary search operator,
an implementation of :meth:`~moptipy.api.operators.Op0.op0`,
to sample exactly one single random solution. It then evaluates the
solution by passing it to
:meth:`~moptipy.api.process.Process.evaluate`. It then terminates.

This is a very very bad optimization algorithm. We only use it in our
book to illustrate one basic concept for solving optimization problems:
The generation of random solutions. The single-random sampling
algorithm here is actually very wasteful: since it only generates
exactly one single solution, it does not use its computational budget
well. Even if you grant it 10'000 years, it will still only generate
one solution. Even if it could generate and test thousands or millions
of solutions, it will not do it. Nevertheless, after applying this
"algorithm," you will have one valid solution remembered in the
optimization process (embodied as instance `process` of
:class:`~moptipy.api.process.Process`).

This concept of random sampling is then refined in the
:class:`~moptipy.algorithms.random_sampling.RandomSampling` as the `rs`
algorithm, which repeats generating random solutions until its allotted
runtime is exhausted.

1. Thomas Weise. *Optimization Algorithms.* 2021. Hefei, Anhui, China:
   Institute of Applied Optimization (IAO), School of Artificial Intelligence
   and Big Data, Hefei University. http://thomasweise.github.io/oa/
"""
from typing import Final

from moptipy.api.algorithm import Algorithm0
from moptipy.api.operators import Op0
from moptipy.api.process import Process


# start book
class SingleRandomSample(Algorithm0):
    """This algorithm creates one single random solution."""

    def solve(self, process: Process) -> None:
        """
        Apply single random sampling to an optimization problem.

        :param process: the black-box process object
        """
        x: Final = process.create()  # Create the solution record.
        self.op0.op0(process.get_random(), x)  # Create random solution
        process.evaluate(x)  # Evaluate that random solution.
# end book

    def __init__(self, op0: Op0) -> None:
        """
        Create the single random sample algorithm.

        :param op0: the nullary search operator
        """
        super().__init__("1rs", op0)
