"""
Select `n` records at random without replacement.

This selection scheme is the standard mating selection scheme in an
Evolutionary Algorithm.
"""

from typing import List, Final, Callable, Any

from numpy.random import Generator

from moptipy.algorithms.modules.selection import Selection, FitnessRecord


class RandomWithoutReplacement(Selection):
    """Select random elements without replacement."""

    def select(self, source: List[FitnessRecord],
               dest: Callable[[FitnessRecord], Any],
               n: int, random: Generator) -> None:
        """
        Select `n` random elements from `source` without replacement.

        :param source: the list with the records to select from
        :param dest: the destination collector to invoke for each selected
            record
        :param n: the number of records to select
        :param random: the random number generator
        """
        source_len: Final[int] = len(source)
        if n == 1:
            dest(source[random.integers(source_len)])
        elif n == 2:
            ri = random.integers
            a = b = ri(source_len)
            while a == b:
                b = ri(source_len)
            dest(source[a])
            dest(source[b])
        else:
            for i in random.choice(source_len, n, False):
                dest(source[i])

    def __str__(self):
        """
        Get the name of the random choice without replacement selection.

        :return: the name of the random choice without replacement selection
            algorithm
        """
        return "rndNoRep"
