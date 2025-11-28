"""
A nullary operator filling an integer list with random values.

>>> from moptipy.utils.nputils import rand_generator
>>> gen = rand_generator(12)

>>> space = IntSpace(4, -2, 2)
>>> op = Op0Random(space)
>>> xx = space.create()

>>> op.op0(gen, xx)
>>> xx
array([-1,  2,  2,  1], dtype=int8)
>>> op.op0(gen, xx)
>>> xx
array([-2, -2, -1, -1], dtype=int8)
>>> op.op0(gen, xx)
>>> xx
array([ 1,  0, -2,  2], dtype=int8)

>>> space = IntSpace(5, 12000, 20000)
>>> op = Op0Random(space)
>>> xx = space.create()

>>> op.op0(gen, xx)
>>> xx
array([15207, 19574, 18014, 12515, 14406], dtype=int16)
>>> op.op0(gen, xx)
>>> xx
array([16080, 13596, 13434, 14148, 16655], dtype=int16)
"""


from typing import Final

import numpy as np
from numpy.random import Generator
from pycommons.types import type_error

from moptipy.api.operators import Op0
from moptipy.spaces.intspace import IntSpace


class Op0Random(Op0):
    """Fill a bit string with random values."""

    def __init__(self, space: IntSpace) -> None:
        """
        Create the Nullary Integer-Space operator.

        :param space: the integer space
        """
        if not isinstance(space, IntSpace):
            raise type_error(space, "space", IntSpace)
        #: the inclusive lower bound
        self.__lb: Final[int] = space.min_value
        #: the exclusive upper bound
        self.__ub: Final[int] = space.max_value + 1

    def op0(self, random: Generator, dest: np.ndarray) -> None:
        """
        Fill the string `dest` with random values.

        :param random: the random number generator
        :param dest: the bit string to fill. Afterward, its contents will
            be random.
        """
        np.copyto(dest, random.integers(
            self.__lb, self.__ub, dest.shape, dest.dtype))

    def __str__(self) -> str:
        """
        Get the name of this operator.

        :return: "uniform"
        """
        return "uniform"
