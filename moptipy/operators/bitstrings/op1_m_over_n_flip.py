"""An operator swapping two elements in a permutation with repetitions."""
from typing import Final, Callable

import numpy as np
from numpy.random import Generator

from moptipy.api import operators
from moptipy.utils.nputils import int_range_to_dtype
from moptipy.utils.types import bool_to_str


class Op1MoverNflip(operators.Op1):
    """
    This unary search operation flips each bit with probability of `m/n`.

    For bit strings of length `n`, draw the number `z` of bits to flip from a
    binomial distribution with `p=m/n`. If `at_least_1` is set to `True`, then
    we repeat drawing `z` until `z>0`.
    """

    def __init__(self, n: int, m: int = 1, at_least_1: bool = True):
        """
        Initialize the operator.

        :param int n: the length of the bit strings
        :param int m: the factor for computing the probability of flipping
            the bits
        :param bool at_least_1: should at least one bit be flipped?
        """
        super().__init__()
        if not isinstance(n, int):
            raise TypeError(f"n must be int but is {type(n)}.")
        if not isinstance(m, int):
            raise TypeError(f"m must be int but is {type(m)}.")
        if not (0 < m <= n):
            raise ValueError(f"m must be in 1..{n}, but is {m}.")
        #: the value of m in p=m/n
        self.__m: Final[int] = m
        if not isinstance(at_least_1, bool):
            raise TypeError(
                f"at_least_1 must be bool but is {type(at_least_1)}.")
        #: is it OK to not flip any bit?
        self.__none_is_ok: Final[bool] = not at_least_1
        #: the internal permutation
        self.__permutation: Final[np.ndarray] = np.array(
            range(n), dtype=int_range_to_dtype(0, n - 1))

    def op1(self, random: Generator, dest: np.ndarray, x: np.ndarray) -> None:
        """
        Copy `x` into `dest` and flip each bit with probability m/n.

        This method will first copy `x` to `dest`. Then it will flip each bit
        in `dest` with probability `m/n`, where `n` is the length of `dest`.
        Regardless of the probability, at least one bit will always be
        flipped if self.at_least_1 is True.

        :param Op1MoverNflip self: the self pointer
        :param Generator random: the random number generator
        :param np.ndarray dest: the array to be shuffled
        :param np.ndarray x: the existing point in the search space
        """
        np.copyto(dest, x)  # copy source to destination
        length: Final[int] = len(dest)  # get n
        p: Final[float] = self.__m / length  # probability to flip bit
        none_is_ok: Final[bool] = self.__none_is_ok

        flips: int  # the number of bits to flip
        rbi: Final[Callable[[int, float], int]] = random.binomial
        while True:
            flips = rbi(length, p)  # compute the number of bits to flip
            if flips > 0:
                break  # we will flip some bit
            if none_is_ok:
                return  # we will flip no bit

        permutation: Final[np.ndarray] = self.__permutation
        i: int = length
        end: Final[int] = length - flips
        ri: Final[Callable[[int], int]] = random.integers
        while i > end:  # we iterate from i=length down to end=length-flips
            k = ri(i)  # get index of next bit index in permutation
            i -= 1  # decrease i
            idx = permutation[k]  # get index of bit to flip and move to end
            permutation[i], permutation[k] = idx, permutation[i]
            dest[idx] = not dest[idx]  # flip bit

    def __str__(self) -> str:
        """
        Get the name of this unary operator.

        :return: "m_over_n_flip"
        :rtype: str
        """
        return f"{self.__m}_over_n_flip_{bool_to_str(not self.__none_is_ok)}"