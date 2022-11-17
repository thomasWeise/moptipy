"""A fitness representing the rank of a solution based on its quality."""

from math import inf

from numpy.random import Generator

from moptipy.algorithms.so.fitness import Fitness, FRecord


# start book
class Rank(Fitness):
    """A fitness computing the rank of an individual based on its quality."""

    def assign_fitness(self, p: list[FRecord], random: Generator) -> None:
        """
        Assign the rank fitness.

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

        rank: int = -1  # the rank counter
        last_fitness: int | float = -inf
        for i, rec in enumerate(p):
            v = rec.fitness
            if v > last_fitness:  # if fitness differs, step rank
                rank = i + 1  # +1 so smallest-possible rank is 1
                last_fitness = v
            rec.fitness = rank  # assign the rank
# end book

    def __str__(self):
        """
        Get the name of this default fitness assignment.

        :return: the name of this fitness assignment strategy
        :retval "rank": always
        """
        return "rank"
