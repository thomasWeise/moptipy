"""
A fitness combining the rank of a solution with the iteration of its creation.

This fitness assignment process is compatible with the simple (mu+lambda)
Evolutionary Algorithm implemented in :class:`~moptipy.algorithms.so.ea.EA`.
It will assign a better fitness to a solution which has a better objective
value. Ties will be broken based on the iteration counter
:attr:`~moptipy.algorithms.so.record.Record.it` of the solution records
:class:`~moptipy.algorithms.so.record.Record`.
"""

from math import inf
from typing import Final

from numpy.random import Generator

from moptipy.algorithms.so.fitness import Fitness, FRecord


# start book
class RankAndIteration(Fitness):
    """
    A fitness joining objective rank and creation iteration.

    The fitness assignment strategy will use two pieces of information to
    determine the fitness of a solution:

    1. the rank the solutions by their objective values
       (:attr:`~moptipy.algorithms.so.record.Record.f`). If two solutions have
       the same fitness, they get the same rank, but the next-worst solution
       will then get a rank with is larger by 2.
    2. the iteration index (:attr:`~moptipy.algorithms.so.record.Record.it`)
       relative to the maximum and minimum iteration index.

    It will multiply the rank of the solution with the range of the iteration
    index in the population and then add the maximum iteration index minus the
    iteration index of the solution, i.e.,

    `fitness(x) = rank(f(x)) * (max_it - min_it + 1)  +  (max_it - it(x))`

    This way, better solutions receive better fitness and ties are broken such
    that younger solutions (with higher iteration index) are preferred.
    In combination with best selection
    (:class:`moptipy.algorithms.modules.selections.best.Best`),
    this replicates the behavior of our simple (mu+lambda) Evolutionary
    Algorithm (:class:`~moptipy.algorithms.so.ea.EA`).
    """

    def assign_fitness(self, p: list[FRecord], random: Generator) -> None:
        """
        Assign the rank and iteration fitness.

        :param p: the list of records
        :param random: ignored
        """
        min_it: int = 9_223_372_036_854_775_808  # minimum iteration index
        max_it: int = -1  # the maximum iteration index

        # In the first iteration, we assign objective value as fitness
        # (for sorting) and get the bounds of the iteration indices.
        for rec in p:  # iterate over list p
            rec.fitness = rec.f  # set f as fitness for sorting
            it: int = rec.it  # get iteration index from record
            if it < min_it:  # update minimum iteration index
                min_it = it
            if it > max_it:  # update maximum iteration index
                max_it = it
        p.sort()  # sort based on objective values

        it_range: Final[int] = max_it - min_it + 1  # range of it index
        rank: int = -1  # the variable for storing the current rank
        last_f: int | float = -inf  # the previous objective value
        for i, rec in enumerate(p):  # iterate over list
            v = rec.fitness  # get the current objective value
            if v > last_f:  # only increase rank if objective f changes
                rank = i + 1  # +1 so smallest-possible fitness is 1
                last_f = v  # remember objective value for comparison
            rec.fitness = (rank * it_range) + max_it - rec.it
# end book

    def __str__(self):
        """
        Get the name of this fitness assignment.

        :return: the name of this fitness assignment strategy
        :retval "rankAndIt": always
        """
        return "rankAndIt"
