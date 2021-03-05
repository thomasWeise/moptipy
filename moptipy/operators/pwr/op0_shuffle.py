"""A nullary operator shuffling a permutation with repetition."""
import numpy as np
from numpy.random import Generator

from moptipy.api import operators


class Op0Shuffle(operators.Op0):
    """Shuffle permutations with repetitions shuffles randomly."""

    def op0(self, random: Generator, dest: np.ndarray) -> None:
        """
        Shuffle the array `dest` randomly.

        :param Generator random: the random number generator
        :param np.ndarray dest: the array to be shuffled
        """
        random.shuffle(dest)

    def get_name(self) -> str:
        """
        Get the name of this operator.

        :return: "shuffle"
        :rtype: str
        """
        return "shuffle"
