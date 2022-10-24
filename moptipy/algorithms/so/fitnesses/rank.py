"""A fitness representing the rank of a solution based on its quality."""

from math import inf
from typing import List, Union

from numpy.random import Generator

from moptipy.algorithms.so.fitness import Fitness, FRecord


class Rank(Fitness):
    """A fitness computing the rank of an individual based on its quality."""

    def assign_fitness(self, p: List[FRecord], random: Generator) -> None:
        """
        Assign the rank fitness.

        :param p: the list of records
        :param random: ignored
        """
        p.sort()  # sort based on objective values

        rank: int = -1  # the rank counter
        last_f: Union[int, float] = -inf
        for rec in p:
            v = rec.fitness
            if v > last_f:  # if fitness differs, step rank
                rank = rank + 1
                last_f = v
            rec.fitness = rank

    def __str__(self):
        """
        Get the name of this default fitness assignment.

        :return: the name of this fitness assignment strategy
        :retval "rank": always
        """
        return "rank"
