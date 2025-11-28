"""
A multi-normal distribution.

>>> from moptipy.utils.nputils import rand_generator
>>> gen = rand_generator(12)

>>> from moptipy.operators.intspace.op0_random import Op0Random
>>> space = IntSpace(4, -2, 2)
>>> op0 = Op0Random(space)
>>> x1 = space.create()
>>> op0.op0(gen, x1)
>>> x1
array([-1,  2,  2,  1], dtype=int8)

>>> op1 = Op1MNormal(space, 1, True, 1.5)
>>> op1.initialize()
>>> x2 = space.create()
>>> op1.op1(gen, x2, x1)
>>> x2
array([-1,  0,  0,  1], dtype=int8)
>>> op1.op1(gen, x2, x1)
>>> x2
array([-1,  2,  1, -2], dtype=int8)
>>> op1.op1(gen, x2, x1)
>>> x2
array([-1,  2,  2,  0], dtype=int8)

>>> space = IntSpace(12, 0, 20)
>>> op0 = Op0Random(space)
>>> x1 = space.create()
>>> op0.op0(gen, x1)
>>> x1
array([ 6,  1, 18, 11, 13, 11,  2,  3, 13,  7, 10, 14], dtype=int8)

>>> op1 = Op1MNormal(space, 1, True, 1.5)
>>> op1.initialize()
>>> x2 = space.create()
>>> op1.op1(gen, x2, x1)
>>> x2
array([ 6,  1, 18, 11, 13, 11,  2,  3, 12,  7,  9, 13], dtype=int8)
>>> op1.op1(gen, x2, x1)
>>> x2
array([ 6,  1, 20, 11, 13, 11,  2,  3, 13,  7, 10, 14], dtype=int8)
>>> op1.op1(gen, x2, x1)
>>> x2
array([ 3,  1, 18, 11, 13, 11,  2,  3, 13,  7,  9, 14], dtype=int8)

>>> str(op1)
'normB1_1d5'
"""
from math import ceil, floor, isfinite
from typing import Final

import numba  # type: ignore
import numpy as np
from numpy.random import Generator
from pycommons.types import check_int_range, type_error

from moptipy.api.operators import Op1
from moptipy.spaces.intspace import IntSpace
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.nputils import (
    fill_in_canonical_permutation,
    int_range_to_dtype,
)
from moptipy.utils.strings import num_to_str_for_name


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def _mnormal(m: int, none_is_ok: bool, permutation: np.ndarray,
             random: Generator, sd: float, min_val: int, max_val: int,
             dest: np.ndarray, x: np.ndarray) -> None:
    """
    Copy `x` into `dest` and sample normal distribution for each element n/m.

    This method will first copy `x` to `dest`. Then it will decide for each
    value in `dest` whether it should be changed: The change happens with
    probability `m/n`, where `n` is the length of `dest`.
    Regardless of the probability, at least one element will always be
    changed if self.at_least_1 is True.

    :param m: the value of m
    :param none_is_ok: is it OK to flip nothing?
    :param permutation: the internal permutation
    :param random: the random number generator
    :param sd: the standard deviation to be used for the normal distribution
    :param min_val: the minimal permissible value
    :param max_val: the maximal permissible value
    :param dest: the destination array to receive the new point
    :param x: the existing point in the search space
    """
    dest[:] = x[:]  # copy source to destination
    length: Final[int] = len(dest)  # get n
    p: Final[float] = m / length  # probability to change values

    flips: int  # the number of values to flip
    while True:
        flips = random.binomial(length, p)  # get number of values to change
        if flips > 0:
            break  # we will change some values
        if none_is_ok:
            return  # we will change no values

    i: int = length
    end: Final[int] = length - flips
    while i > end:  # we iterate from i=length down to end=length-change
        k = random.integers(0, i)  # index of next value index in permutation
        i -= 1  # decrease i
        idx = permutation[k]  # get index of bit to value and move to end
        permutation[i], permutation[k] = idx, permutation[i]

        # put a normal distribution around old value and sample
        # a new value
        old_value = dest[idx]
        while True:
            rnd = random.normal(scale=sd)
            new_value = old_value + (ceil(rnd) if rnd > 0 else floor(rnd))
            if (new_value != old_value) and (min_val <= new_value <= max_val):
                break
        dest[idx] = new_value


class Op1MNormal(Op1):
    """Randomly choose a number of ints to change with normal distribution."""

    def __init__(self, space: IntSpace, m: int = 1, at_least_1: bool = True,
                 sd: float = 1.0):
        """
        Initialize the operator.

        :param n: the length of the bit strings
        :param m: the factor for computing the probability of flipping
            the bits
        :param at_least_1: should at least one bit be flipped?
        """
        super().__init__()
        if not isinstance(space, IntSpace):
            raise type_error(space, "space", IntSpace)
        #: the internal dimension
        self.__n: Final[int] = space.dimension
        #: the minimum permissible value
        self.__min: Final[int] = space.min_value
        #: the maximum permissible value
        self.__max: Final[int] = space.max_value
        #: the value of m in p=m/n
        self.__m: Final[int] = check_int_range(m, "m", 1, self.__n)
        if not isinstance(at_least_1, bool):
            raise type_error(at_least_1, "at_least_1", bool)
        #: is it OK to not flip any bit?
        self.__none_is_ok: Final[bool] = not at_least_1
        #: the internal permutation
        self.__permutation: Final[np.ndarray] = np.empty(
            self.__n, dtype=int_range_to_dtype(0, self.__n - 1))
        if not isinstance(sd, float):
            raise type_error(sd, "sd", float)
        if not (isfinite(sd) and (0 < sd <= (self.__max - self.__min + 1))):
            raise ValueError(
                f"Invalid value {sd} for sd with {self.__min}..{self.__max}.")
        #: the internal standard deviation
        self.__sd: Final[float] = sd

    def initialize(self) -> None:
        """Initialize this operator."""
        super().initialize()
        fill_in_canonical_permutation(self.__permutation)

    def op1(self, random: Generator, dest: np.ndarray, x: np.ndarray) -> None:
        """
        Copy `x` into `dest` and change each value with probability m/n.

        This method will first copy `x` to `dest`. Then it will change each
        value in `dest` with probability `m/n`, where `n` is the length of
        `dest`. Regardless of the probability, at least one value will always
        be changed if self.at_least_1 is True.

        If a value is changed, we will put a normal distribution around it and
        sample it from that. Of course, we only accept values within the
        limits of the integer space and that are different from the original
        value.

        :param self: the self pointer
        :param random: the random number generator
        :param dest: the destination array to receive the new point
        :param x: the existing point in the search space
        """
        _mnormal(self.__m, self.__none_is_ok, self.__permutation, random,
                 self.__sd, self.__min, self.__max, dest, x)

    def __str__(self) -> str:
        """
        Get the name of this unary operator.

        :return: "fileB" + m + "n" if none-is-ok else ""
        """
        return (f"normB{self.__m}{'n' if self.__none_is_ok else ''}_"
                f"{num_to_str_for_name(self.__sd)}")

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of this operator to the given logger.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        logger.key_value("m", self.__m)
        logger.key_value("n", self.__n)
        logger.key_value("sd", self.__sd)
        logger.key_value("min", self.__min)
        logger.key_value("max", self.__max)
