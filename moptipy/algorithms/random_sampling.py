"""
The random sampling algorithm `rs`.

The random sampling algorithm keeps creating random solutions until
the computational budget is exhausted or any other termination
criterion is met.
In a loop that is repeated until the termination criterion
:meth:`~moptipy.api.process.Process.should_terminate` becomes `True`,
it

- applies the nullary search operator, an implementation of
  :meth:`~moptipy.api.operators.Op0.op0`, to sample exactly one single
  random solution. It then
- evaluates the solution by passing it to
  :meth:`~moptipy.api.process.Process.evaluate`.

This algorithm has a very bad performance. It makes no use of the
information that is learned during the search. Every single solution is
created completely independent on anything that has happened before
in the algorithm. Only if the problem is completely random, has no
structure, or is entirely deceptive, this algorithm can be of any use.

Basically, this algorithm is an iterated version of the `1rs` algorithm
implemented in
:class:`~moptipy.algorithms.single_random_sample.SingleRandomSample`.

1. Thomas Weise. *Optimization Algorithms.* 2021. Hefei, Anhui, China:
   Institute of Applied Optimization (IAO), School of Artificial Intelligence
   and Big Data, Hefei University. http://thomasweise.github.io/oa/
"""
from typing import Callable, Final

from numpy.random import Generator

from moptipy.api.algorithm import Algorithm0
from moptipy.api.operators import Op0
from moptipy.api.process import Process


# start book
class RandomSampling(Algorithm0):
    """In each step, random sampling creates a new, random solution."""

    def solve(self, process: Process) -> None:
        """
        Apply the random sampling approach to an optimization problem.

        :param process: the black-box process object
        """
        x: Final = process.create()  # Create the solution record.
        # Obtain the random number generator.
        random: Final[Generator] = process.get_random()

        # Put function references in variables to save time.
        evaluate: Final[Callable] = process.evaluate  # the objective
        op0: Final[Callable] = self.op0.op0  # the nullary operator
        should_terminate: Final[Callable] = process.should_terminate

        while not should_terminate():  # Until we need to quit...
            op0(random, x)  # Sample a completely random solution.
            evaluate(x)  # Evaluate the solution ... but ignore result.
# end book

    def __init__(self, op0: Op0) -> None:
        """
        Create the random sampling algorithm.

        :param op0: the nullary search operator
        """
        super().__init__("rs", op0)
