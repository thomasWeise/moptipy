from moptipy.api import operators
from numpy.random import Generator
import numpy as np


class Op0(operators.Op0):
    """
    The nullary search operation for permutations with repetitions shuffles
    them randomly.
    """

    def op0(self, random: Generator, dest: np.ndarray):
        """
        Shuffle the array `dest` randomly.
        :param random: the random number generator
        :param dest: the array to be shuffled
        """
        random.shuffle(dest)

    def get_name(self) -> str:
        return "shuffle"
