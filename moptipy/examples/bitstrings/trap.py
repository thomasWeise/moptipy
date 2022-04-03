"""The well-known Trap problem."""
from typing import Final

import numba  # type: ignore
import numpy as np

from moptipy.api.objective import Objective


@numba.njit(nogil=True, cache=True)
def trap(x: np.ndarray) -> int:
    """
    Compute the trap objective value.

    :param x: the np array
    :return: the trap function value

    >>> print(trap(np.array([True, True, False, False, False])))
    3
    >>> print(trap(np.array([True, False, True, False, False])))
    3
    >>> print(trap(np.array([False, True,  True, False, False])))
    3
    >>> print(trap(np.array([True, True, True, True, True])))
    0
    >>> print(trap(np.array([False, True, True, True, True])))
    5
    >>> print(trap(np.array([False, False, False, False, False])))
    1
    >>> print(trap(np.array([False, True,  False, False, False])))
    2
    >>> print(trap(np.array([False, True,  True, True, False])))
    4
    """
    l: Final[int] = len(x)
    s: Final[int] = x.sum()
    return 0 if (s >= l) else int(s + 1)


class Trap(Objective):
    """The trap problem."""

    def __init__(self, n: int) -> None:  # +book
        """
        Initialize the trap objective function.

        :param n: the dimension of the problem

        >>> print(Trap(55).n)
        55
        >>> print(Trap(4).evaluate(np.array([True, True, False, True])))
        4
        """
        super().__init__()
        if not isinstance(n, int):
            raise TypeError(
                f"Must provide integer n, but got '{type(n)}'.")
        #: the upper bound = the length of the bit strings
        self.n: Final[int] = n
        self.evaluate = trap  # type: ignore

    def lower_bound(self) -> int:
        """
        Get the lower bound of the trap objective function.

        :return: 0

        >>> print(Trap(20).lower_bound())
        0
        """
        return 0

    def upper_bound(self) -> int:
        """
        Get the upper bound of the trap objective function.

        :return: the length of the bit string

        >>> print(Trap(40).upper_bound())
        40
        """
        return self.n

    def __str__(self) -> str:
        """
        Get the name of the trap objective function.

        :return: `trap_` + length of string

        >>> print(Trap(33))
        trap_33
        """
        return f"trap_{self.n}"
