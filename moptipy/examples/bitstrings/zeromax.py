"""
An objective function counting the number of zeros in a bit string.

This problem exists mainly for testing purposes as counterpart of
:class:`~moptipy.examples.bitstrings.onemax.OneMax`.
"""
from typing import Final

import numba  # type: ignore
import numpy as np

from moptipy.api.objective import Objective
from moptipy.utils.types import type_error


@numba.njit(nogil=True, cache=True)
def zeromax(x: np.ndarray) -> int:
    """
    Get the length of a string minus the number of zeros in it.

    :param x: the np array
    :return: the length of the string minus the number of zeros, i.e., the
        number of ones

    >>> print(zeromax(np.array([True, True, False, False, False])))
    2
    >>> print(zeromax(np.array([True, False, True, False, False])))
    2
    >>> print(zeromax(np.array([False, True,  True, False, False])))
    2
    >>> print(zeromax(np.array([True, True, True, True, True])))
    5
    >>> print(zeromax(np.array([False, True, True, True, True])))
    4
    >>> print(zeromax(np.array([False, False, False, False, False])))
    0
    """
    return int(x.sum())


class ZeroMax(Objective):
    """Maximize the number of zeros in a bit string."""

    def __init__(self, n: int) -> None:
        """
        Initialize the zeromax objective function.

        :param n: the dimension of the problem

        >>> print(ZeroMax(2).n)
        2
        >>> print(ZeroMax(4).evaluate(np.array([True, True, False, True])))
        3
        """
        super().__init__()
        if not isinstance(n, int):
            raise type_error(n, "n", int)
        #: the upper bound = the length of the bit strings
        self.n: Final[int] = n
        self.evaluate = zeromax  # type: ignore

    def lower_bound(self) -> int:
        """
        Get the lower bound of the zeromax objective function.

        :return: 0

        >>> print(ZeroMax(10).lower_bound())
        0
        """
        return 0

    def upper_bound(self) -> int:
        """
        Get the upper bound of the zeromax objective function.

        :return: the length of the bit string

        >>> print(ZeroMax(7).upper_bound())
        7
        """
        return self.n

    def is_always_integer(self) -> bool:
        """
        Return `True` because :func:`zeromax` always returns `int` values.

        :retval True: always
        """
        return True

    def __str__(self) -> str:
        """
        Get the name of the zeromax objective function.

        :return: `zeromax_` + length of string

        >>> print(ZeroMax(13))
        zeromax_13
        """
        return f"zeromax_{self.n}"
