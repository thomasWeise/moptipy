"""
The implementation of the Randomized Local Search algorithm `rls`.

The algorithm starts by applying the nullary search operator, an
implementation of :meth:`~moptipy.api.operators.Op0.op0`, to sample
one fully random solution. This is the first best-so-far solution.
In each step, it applies the unary operator, an implementation of
:meth:`~moptipy.api.operators.Op1.op1`, to the best-so-far solution to
obtain a new, similar solution. If this new solution is not worse than
the current best-so-far solution, it replaces this solution. Otherwise,
it is discarded.

The `rls` algorithm is a simple local search that accepts all
non-deteriorating moves. It is thus similar to the simple hill climber
`hc` implemented in
:class:`moptipy.algorithms.hill_climber.HillClimber`, which, however,
accepts strictly improving moves.

Strictly speaking, the name "Randomized Local Search" only fits
partially to the algorithm we implement here. Take the discrete search
domain, where the search spaces are bit strings of a fixed length `n`,
as an example. The name "Randomized Local Search" and the abbreviation
`rls` has a fixed meaning on this domain: It is the algorithm that
starts with a random solution and flips exactly one randomly chosen bit
in each step. This corresponds to our `rls` algorithm with the operator
:class:`~moptipy.operators.bitstrings.op1_flip1.Op1Flip1`. However,
an algorithm that starts with a random solution and flips a number of
bits sampled from a Binomial distribution is called `(1+1) EA`. Now
this algorithm corresponds again to our `rls` algorithm, but this time
with operator
:class:`~moptipy.operators.bitstrings.op1_m_over_n_flip.Op1MoverNflip`.
In other words, we can implement (at least) two algorithms with
well-known and fixed names by plugging different operators into our
`rls` approach. One of them is called `RLS`, the other one is called
`(1+1) EA`. Now this is somewhat confusing but results from the general
nature of our basic framework. Regardless of what we do, we will have
some form of name clash here. We advise the user of our algorithms to
be careful with respect to literature and scientific conventions when
using our framework.
"""
from typing import Final, Union, Callable

from numpy.random import Generator

from moptipy.api.algorithm import Algorithm1
from moptipy.api.process import Process


# start book
class RLS(Algorithm1):
    """
    The RLS is a simple local search accepting all non-worsening moves.

    In each step, an RLS creates a modified copy `new_x` of the
    current best solution `best_x`. If `new_x` is not worse than `best_x`,
    it becomes the new `best_x`. Otherwise, it is discarded.
    """

    def solve(self, process: Process) -> None:
        """
        Apply the RLS to an optimization problem.

        :param process: the black-box process object
        """
        # Create records for old and new point in the search space.
        best_x = process.create()  # record for best-so-far solution
        new_x = process.create()  # record for new solution
        # Obtain the random number generator.
        random: Final[Generator] = process.get_random()

        # Put function references in variables to save time.
        evaluate: Final[Callable] = process.evaluate  # the objective
        op1: Final[Callable] = self.op1.op1  # the unary operator
        should_terminate: Final[Callable] = process.should_terminate

        # Start at a random point in the search space and evaluate it.
        self.op0.op0(random, best_x)  # Create 1 solution randomly and
        best_f: Union[int, float] = evaluate(best_x)  # evaluate it.

        while not should_terminate():  # Until we need to quit...
            op1(random, new_x, best_x)  # new_x = neighbor of best_x
            new_f: Union[int, float] = evaluate(new_x)
            if new_f <= best_f:  # new_x is not worse than best_x?
                best_f = new_f  # Store its objective value.
                best_x, new_x = new_x, best_x  # Swap best and new.
# end book

    def __str__(self) -> str:
        """
        Get the name of this RLS algorithm.

        :return: "rls" + any non-standard operator suffixes
        """
        return f"rls{super().__str__()}"
