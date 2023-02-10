"""
A fitness representing the rank of a solution based on its quality.

First, all solutions are sorted based on their objective values. Then, they
receive their rank, i.e., index+1 in the sorted list, as fitness. If two
elements have the same objective value, they also receive the same rank.
This rank will be the index+1 of the first element.

>>> # The first value in FRecord(x, y) is the solution, which is not matter
>>> # here. The second value is the objective value used for ranking.
>>> l = [FRecord(1, 10), FRecord(2, 3), FRecord(3, 3), FRecord(4, 2)]
>>> from numpy.random import default_rng
>>> Rank().assign_fitness(l, default_rng())
>>> for z in l:
...     print(f"x={z.x}, fitness={z.fitness}")
x=4, fitness=1
x=2, fitness=2
x=3, fitness=2
x=1, fitness=4

Together with fitness proportionate selection (:class:`~moptipy.algorithms.\
modules.selections.fitness_proportionate_sus.FitnessProportionateSUS`), it
works very similar to linear ranking selection in an EA
(:class:`~moptipy.algorithms.so.general_ea.GeneralEA`).

1. L. Darrell Whitley. The GENITOR Algorithm and Selection Pressure: Why
   Rank-Based Allocation of Reproductive Trials is Best. In J. David Schaffer,
   ed., Proceedings of the 3rd International Conference on Genetic Algorithms
   (ICGA'89), June 4-7, 1989, Fairfax, VA, USA, pages 116-121. San Francisco,
   CA, USA: Morgan Kaufmann Publishers Inc. ISBN: 1-55860-066-3
   https://www.researchgate.net/publication/2527551
2. Tobias Blickle and Lothar Thiele. A Comparison of Selection Schemes used in
   Genetic Algorithms. Second edition, December 1995. TIK-Report 11 from the
   Eidgenössische Technische Hochschule (ETH) Zürich, Department of Electrical
   Engineering, Computer Engineering and Networks Laboratory (TIK), Zürich,
   Switzerland. ftp://ftp.tik.ee.ethz.ch/pub/publications/TIK-Report11.ps
"""

from math import inf

from numpy.random import Generator

from moptipy.algorithms.so.fitness import Fitness, FRecord


# start book
class Rank(Fitness):
    """A fitness computing the rank of an individual based on its quality."""

    def assign_fitness(self, p: list[FRecord], random: Generator) -> None:
        """
        Assign the rank as fitness.

        :param p: the list of records
        :param random: ignored

        >>> l = [FRecord(0, 10), FRecord(1, 5), FRecord(2, 5), FRecord(3, -1)]
        >>> from numpy.random import default_rng
        >>> Rank().assign_fitness(l, default_rng())
        >>> l[0].x
        3
        >>> l[0].fitness
        1
        >>> l[1].x
        1
        >>> l[1].fitness
        2
        >>> l[2].x
        2
        >>> l[2].fitness
        2
        >>> l[3].x
        0
        >>> l[3].fitness
        4
        """
        for rec in p:  # first copy rec.f to rec.fitness
            rec.fitness = rec.f  # because then we can easily sort
        p.sort()  # sort based on objective values

        rank: int = -1  # the variable for storing the current rank
        last_f: int | float = -inf  # the previous objective value
        for i, rec in enumerate(p):  # iterate over list
            v = rec.fitness  # get the current objective value
            if v > last_f:  # only increase rank if objective f changes
                rank = i + 1  # +1 so smallest-possible fitness is 1
                last_f = v  # remember objective value for comparison
            rec.fitness = rank  # assign the rank (same f = same rank)
# end book

    def __str__(self):
        """
        Get the name of this rank-based fitness assignment.

        :return: the name of this fitness assignment strategy
        :retval "rank": always
        """
        return "rank"
