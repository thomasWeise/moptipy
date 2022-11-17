"""
Select `n` records at random without replacement.

This selection scheme is the standard mating selection scheme in an
Evolutionary Algorithm.
"""

from typing import Any, Callable, Final

from numpy.random import Generator

from moptipy.algorithms.modules.selection import FitnessRecord, Selection


# start book
class RandomWithoutReplacement(Selection):
    """Select random elements without replacement."""

    def select(self, source: list[FitnessRecord],
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
        m: Final[int] = len(source)
        if n == 1:  # handle n=1 exactly as in (mu+lambda) EA
            dest(source[random.integers(m)])  # pick 1 solution randomly
        elif n == 2:  # handle n=2 exactly as in (mu+lambda) EA
            ri = random.integers  # fast call
            a = b = ri(m)  # get first random index
            while a == b:  # find a second, different random index
                b = ri(m)
            dest(source[a])  # send first solution to dest
            dest(source[b])  # send second solution to dest
        else:  # handle other cases: n ints from 0..m-1 w/o replacement
            for i in random.choice(m, n, False):  # get the ints
                dest(source[i])  # send randomly chosen records to dest
# end book

    def __str__(self):
        """
        Get the name of the random choice without replacement selection.

        :return: the name of the random choice without replacement selection
            algorithm
        """
        return "rndNoRep"
