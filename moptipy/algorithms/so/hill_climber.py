"""
The implementation of the basic hill climbing algorithm `hc`.

The algorithm starts by applying the nullary search operator, an
implementation of :meth:`~moptipy.api.operators.Op0.op0`, to sample one fully
random solution. This is the first best-so-far solution. In each step, it
applies the unary operator, an implementation of
:meth:`~moptipy.api.operators.Op1.op1`, to the best-so-far solution to obtain
a new, similar solution. If this new solution is strictly better than the
current best-so-far solution, it replaces this solution. Otherwise, it is
discarded.

The hill climbing algorithm is a simple local search that only accepts
strictly improving moves. It is thus similar to the randomized local search
(`rls`) implemented in :class:`~moptipy.algorithms.so.rls.RLS`, which, however,
accepts non-deteriorating moves. We also provide `hcr`, a variant of the hill
climber that restarts automatically with a certain number of moves were not
able to improve the current best-so-far solution in class :class:`~moptipy.\
algorithms.so.hill_climber_with_restarts.HillClimberWithRestarts`.

1. Stuart Jonathan Russell and Peter Norvig. *Artificial Intelligence: A
   Modern Approach (AIMA)*. 2nd edition. 2002. Upper Saddle River, NJ, USA:
   Prentice Hall International Inc. ISBN: 0-13-080302-2
2. Steven S. Skiena. *The Algorithm Design Manual.* 2nd edition. 2008.
   London, UK: Springer-Verlag. ISBN: 978-1-84800-069-8.
   http://doi.org/10.1007/978-1-84800-070-4.
3. David Stifler Johnson, Christos H. Papadimitriou, and Mihalis Yannakakis.
   How Easy Is Local Search? Journal of Computer and System Sciences.
   37(1):79-100. August 1988. http://doi.org/10.1016/0022-0000(88)90046-3
   http://www2.karlin.mff.cuni.cz/~krajicek/jpy2.pdf
4. James C. Spall. *Introduction to Stochastic Search and Optimization.*
   April 2003. Estimation, Simulation, and Control -- Wiley-Interscience
   Series in Discrete Mathematics and Optimization, volume 6. Chichester, West
   Sussex, UK: Wiley Interscience. ISBN: 0-471-33052-3.
   http://www.jhuapl.edu/ISSO/
5. Holger H. Hoos and Thomas Stützle. *Stochastic Local Search: Foundations
   and Applications.* 2005. ISBN: 1493303732. In The Morgan Kaufmann Series in
   Artificial Intelligence. Amsterdam, The Netherlands: Elsevier.
6. Thomas Weise. *Optimization Algorithms.* 2021. Hefei, Anhui, China:
   Institute of Applied Optimization (IAO), School of Artificial Intelligence
   and Big Data, Hefei University. http://thomasweise.github.io/oa/
7. Thomas Weise. *Global Optimization Algorithms - Theory and Application.*
   2009. http://www.it-weise.de/projects/book.pdf
"""
from typing import Callable, Final

from numpy.random import Generator

from moptipy.api.algorithm import Algorithm1
from moptipy.api.operators import Op0, Op1
from moptipy.api.process import Process


# start book
class HillClimber(Algorithm1):
    """
    The stochastic hill climbing algorithm only accepts improving moves.

    In each step, a hill climber creates a modified copy `new_x` of the
    current best solution `best_x`. If `new_x` is better than `best_x`,
    it becomes the new `best_x`. Otherwise, it is discarded.
    """

    def solve(self, process: Process) -> None:
        """
        Apply the hill climber to an optimization problem.

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
        best_f: int | float = evaluate(best_x)  # evaluate it.

        while not should_terminate():  # Until we need to quit...
            op1(random, new_x, best_x)  # new_x = neighbor of best_x
            new_f: int | float = evaluate(new_x)
            if new_f < best_f:  # new_x is _better_ than best_x?
                best_f = new_f  # Store its objective value.
                best_x, new_x = new_x, best_x  # Swap best and new.
# end book

    def __init__(self, op0: Op0, op1: Op1) -> None:
        """
        Create the hill climber.

        :param op0: the nullary search operator
        :param op1: the unary search operator
        """
        super().__init__("hc", op0, op1)
