"""An objective function counting the leading ones in a bit string."""
from typing import Final

import numba  # type: ignore
import numpy as np

from moptipy.api.objective import Objective
from moptipy.utils.types import type_error


@numba.njit(nogil=True, cache=True)
def leadingones(x: np.ndarray) -> int:
    """
    Get the length of the string minus the number of leading ones.

    :param x: the np array
    :return: the number of leading ones

    >>> print(leadingones(np.array([False, False, True, False, False])))
    5
    >>> print(leadingones(np.array([True, False, False, True, True])))
    4
    >>> print(leadingones(np.array([True, True, False, False, False])))
    3
    >>> print(leadingones(np.array([True, True, True, False, True])))
    2
    >>> print(leadingones(np.array([True, True, True, True, False])))
    1
    >>> print(leadingones(np.array([True, True, True, True, True])))
    0
    """
    xlen: Final[int] = len(x)
    for i in range(xlen):
        if not x[i]:
            return xlen - i
    return 0


class LeadingOnes(Objective):
    """Maximize the number of leadings ones in a bit string."""

    def __init__(self, n: int) -> None:  # +book
        """
        Initialize the leading ones objective function.

        :param n: the dimension of the problem

        >>> print(LeadingOnes(55).n)
        55
        >>> print(LeadingOnes(4).evaluate(np.array([True, True, True, False])))
        1
        """
        super().__init__()
        if not isinstance(n, int):
            raise type_error(n, "n", int)
        #: the upper bound = the length of the bit strings
        self.n: Final[int] = n
        self.evaluate = leadingones  # type: ignore

    def lower_bound(self) -> int:
        """
        Get the lower bound of the leadingones objective function.

        :return: 0

        >>> print(LeadingOnes(3).lower_bound())
        0
        """
        return 0

    def upper_bound(self) -> int:
        """
        Get the upper bound of the leadingones objective function.

        :return: the length of the bit string

        >>> print(LeadingOnes(5).upper_bound())
        5
        """
        return self.n

    def is_always_integer(self) -> bool:
        """
        Return `True` because :func:`leadingones` always returns `int` values.

        :retval True: always
        """
        return True

    def __str__(self) -> str:
        """
        Get the name of the leadingones objective function.

        :return: `leadingones_` + lenth of string

        >>> print(LeadingOnes(10))
        leadingones_10
        """
        return f"leadingones_{self.n}"
