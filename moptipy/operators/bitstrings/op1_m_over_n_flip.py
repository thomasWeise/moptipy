"""A unary operator flipping each bit with probability m/n."""
from typing import Callable, Final

import numpy as np
from numpy.random import Generator

from moptipy.api.operators import Op1
from moptipy.utils.nputils import (
    fill_in_canonical_permutation,
    int_range_to_dtype,
)
from moptipy.utils.types import check_int_range, type_error


class Op1MoverNflip(Op1):
    """
    A unary search operation that flips each bit with probability of `m/n`.

    For bit strings of length `n`, draw the number `z` of bits to flip from a
    binomial distribution with `p=m/n`. If `at_least_1` is set to `True`, then
    we repeat drawing `z` until `z>0`.
    """

    def __init__(self, n: int, m: int = 1, at_least_1: bool = True):
        """
        Initialize the operator.

        :param n: the length of the bit strings
        :param m: the factor for computing the probability of flipping
            the bits
        :param at_least_1: should at least one bit be flipped?
        """
        super().__init__()
        check_int_range(n, "n", 1)
        #: the value of m in p=m/n
        self.__m: Final[int] = check_int_range(m, "m", 1, n)
        if not isinstance(at_least_1, bool):
            raise type_error(at_least_1, "at_least_1", bool)
        #: is it OK to not flip any bit?
        self.__none_is_ok: Final[bool] = not at_least_1
        #: the internal permutation
        self.__permutation: Final[np.ndarray] = np.empty(
            n, dtype=int_range_to_dtype(0, n - 1))

    def initialize(self) -> None:
        """Initialize this operator."""
        super().initialize()
        fill_in_canonical_permutation(self.__permutation)

    def op1(self, random: Generator, dest: np.ndarray, x: np.ndarray) -> None:
        """
        Copy `x` into `dest` and flip each bit with probability m/n.

        This method will first copy `x` to `dest`. Then it will flip each bit
        in `dest` with probability `m/n`, where `n` is the length of `dest`.
        Regardless of the probability, at least one bit will always be
        flipped if self.at_least_1 is True.

        :param self: the self pointer
        :param random: the random number generator
        :param dest: the destination array to receive the new point
        :param x: the existing point in the search space
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
        """
        return f"flipB{self.__m}{'n' if self.__none_is_ok else ''}"
