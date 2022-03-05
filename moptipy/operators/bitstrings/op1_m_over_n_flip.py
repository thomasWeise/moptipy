"""An operator swapping two elements in a permutation with repetitions."""
from typing import Final

import numpy as np
from numpy.random import Generator

from moptipy.api import operators
from moptipy.utils.types import bool_to_str


class Op1MoverNflip(operators.Op1):
    """This unary search operation flips each bit with probability of m/n."""

    def __init__(self, m: int = 1, at_least_1: bool = True):
        """
        Initialize the operator.

        :param int m: the factor for computing the probability of flipping
            the bits
        :param bool at_least_1: should at least one bit be flipped?
        """
        super().__init__()
        if not isinstance(m, int):
            raise TypeError(f"m must be int but is {type(m)}.")
        if m < 0:
            raise ValueError(f"m must be > 0, but is {m}.")
        self.m: Final[int] = m
        if not isinstance(at_least_1, bool):
            raise TypeError(
                f"at_least_1 must be bool but is {type(at_least_1)}.")
        self.at_least_1: Final[bool] = at_least_1

    def op1(self, random: Generator, dest: np.ndarray, x: np.ndarray) -> None:
        """
        Copy `x` into `dest` and flip each bit with probability m/n.

        This method will first copy `x` to `dest`. Then it will flip each bit
        in `dest` with probability `m/n`, where `n` is the length of `dest`.
        Regardless of the probability, at least one bit will always be
        flipped if self.at_least_1 is True.

        :param Generator random: the random number generator
        :param np.ndarray dest: the array to be shuffled
        :param np.ndarray x: the existing point in the search space
        """
        np.copyto(dest, x)
        length: Final[int] = len(dest)
        mm: Final[int] = self.m
        repeat: bool = True

        while repeat:
            repeat = self.at_least_1
            for i in range(length):
                if random.integers(length) < mm:
                    dest[i] = not dest[i]
                    repeat = False

    def __str__(self) -> str:
        """
        Get the name of this unary operator.

        :return: "m_over_n_flip"
        :rtype: str
        """
        return f"{self.m}_over_n_flip_{bool_to_str(self.at_least_1)}"
