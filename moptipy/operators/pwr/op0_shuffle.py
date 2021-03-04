"""
In this module, we implement a nullary operator shuffling the contents of a
permutation with repetition randomly.
"""
import numpy as np
from numpy.random import Generator

from moptipy.api import operators


class Op0Shuffle(operators.Op0):
    """
    The nullary search operation for permutations with repetitions shuffles
    them randomly.
    """

    def op0(self, random: Generator, dest: np.ndarray) -> None:
        """
        Shuffle the array `dest` randomly.
        :param random: the random number generator
        :param dest: the array to be shuffled
        """
        random.shuffle(dest)

    def get_name(self) -> str:
        return "shuffle"
