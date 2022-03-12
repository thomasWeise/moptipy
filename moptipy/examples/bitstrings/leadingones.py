"""An objective function counting the leading ones in a bit string."""
from typing import Final

import numba  # type: ignore
import numpy as np

from moptipy.api.objective import Objective


@numba.njit(nogil=True, cache=True)
def leadingones(x: np.ndarray) -> int:
    """
    Get the length of the string minus the number of leading ones.

    :param np.ndarray x: the np array
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

    def __init__(self, dimension: int) -> None:  # +book
        """
        Initialize the leading ones objective function.

        :param int dimension: the dimension of the problem
        """
        super().__init__()
        if not isinstance(dimension, int):
            raise TypeError(
                f"Must provide int dimension, but got '{type(dimension)}'.")
        #: the upper bound = the length of the bit strings
        self.__ub: Final[int] = dimension
        self.evaluate = leadingones  # type: ignore

    def lower_bound(self) -> int:
        """
        Get the lower bound of the leadingones objective function.

        :return: 0
        :rtype: int
        """
        return 0

    def upper_bound(self) -> int:
        """
        Get the upper bound of the leadingones objective function.

        :return: the length of the bit string
        :rtype: int
        """
        return self.__ub

    def __str__(self) -> str:
        """
        Get the name of the leadingones objective function.

        :return: `leadingones_` + lenth of string
        :rtype: str
        """
        return f"leadingones_{self.__ub}"
