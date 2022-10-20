"""
Truncation selection chooses the `n` best elements from the source array.

Truncation selection is the standard way to perform survival selection in an
(mu+lambda) Evolutionary Algorithm.
"""

from typing import List

from numpy.random import Generator


from moptipy.algorithms.modules.selection import Selection, FitnessRecord


class Truncation(Selection):
    """The truncation selections: select the best `n` elements."""

    def select(self, source: List[FitnessRecord], dest: List[FitnessRecord],
               n: int, random: Generator) -> None:  # pylint: disable=W0613
        """
        Perform truncation selections.

        :param source: the list with the records to select from
        :param dest: the destination to append the selected records to
        :param n: the number of records to select
        :param random: the random number generator
        """
        source.sort()
        dest.extend(source[0:n])

    def __str__(self):
        """
        Get the name of the truncation selection algorithm.

        :return: the name of the truncation selection algorithm
        """
        return "trunc"
