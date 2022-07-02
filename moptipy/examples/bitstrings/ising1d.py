"""The one-dimensional Ising problem."""
from typing import Final

import numba  # type: ignore
import numpy as np

from moptipy.api.objective import Objective
from moptipy.utils.types import type_error


@numba.njit(nogil=True, cache=True)
def ising1d(x: np.ndarray) -> int:
    """
    Compute the objective value of the 1-dimensional Ising problem.

    :param x: the np array
    :return: the trap function value

    >>> print(ising1d(np.array([True, True, True, True, True])))
    0
    >>> print(ising1d(np.array([False, False, False, False, False])))
    0
    >>> print(ising1d(np.array([False, False, False, True, False])))
    2
    >>> print(ising1d(np.array([True, False, False, False, False])))
    2
    >>> print(ising1d(np.array([False, False, False, False, True])))
    2
    >>> print(ising1d(np.array([True, False, False, False, True])))
    2
    >>> print(ising1d(np.array([True, False, True, False, False])))
    4
    >>> print(ising1d(np.array([True, False, True, False, True, False])))
    6
    """
    s: int = 0
    prev: bool = x[-1]
    for cur in x:
        if cur != prev:
            s += 1
        prev = cur
    return s


class Ising1d(Objective):
    """The one-dimensional Ising problem."""

    def __init__(self, n: int) -> None:  # +book
        """
        Initialize the one-dimensional Ising problem.

        :param n: the dimension of the problem

        >>> print(Ising1d(7).n)
        7
        >>> print(Ising1d(3).evaluate(np.array([True, False, True])))
        2
        """
        super().__init__()
        if not isinstance(n, int):
            raise type_error(n, "n", int)
        #: the upper bound = the length of the bit strings
        self.n: Final[int] = n
        self.evaluate = ising1d  # type: ignore

    def lower_bound(self) -> int:
        """
        Get the lower bound of the one-dimensional Ising problem.

        :return: 0

        >>> print(Ising1d(7).lower_bound())
        0
        """
        return 0

    def upper_bound(self) -> int:
        """
        Get the upper bound of the one-dimensional Ising problem.

        :return: the length of the bit string

        >>> print(Ising1d(12).upper_bound())
        12
        """
        return self.n

    def is_always_integer(self) -> bool:
        """
        Return `True` because :func:`ising1d` always returns `int` values.

        :retval True: always
        """
        return True

    def __str__(self) -> str:
        """
        Get the name of the one-dimensional Ising problem.

        :return: `ising1d_` + length of string

        >>> print(Ising1d(5))
        ising1d_5
        """
        return f"ising1d_{self.n}"
