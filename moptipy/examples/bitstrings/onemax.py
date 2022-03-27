"""An objective function counting the number of ones in a bit string."""
from typing import Final

import numba  # type: ignore
import numpy as np

from moptipy.api.objective import Objective


@numba.njit(nogil=True, cache=True)
def onemax(x: np.ndarray) -> int:
    """
    Get the length of a string minus the number of ones in it.

    :param np.ndarray x: the np array
    :return: the number of ones

    >>> print(onemax(np.array([True, True, False, False, False])))
    3
    >>> print(onemax(np.array([True, False, True, False, False])))
    3
    >>> print(onemax(np.array([False, True,  True, False, False])))
    3
    >>> print(onemax(np.array([True, True, True, True, True])))
    0
    >>> print(onemax(np.array([False, True, True, True, True])))
    1
    >>> print(onemax(np.array([False, False, False, False, False])))
    5
    """
    return int(len(x) - x.sum())


class OneMax(Objective):
    """Maximize the number of ones in a bit string."""

    def __init__(self, n: int) -> None:  # +book
        """
        Initialize the onemax objective function.

        :param int n: the dimension of the problem

        >>> print(OneMax(2).n)
        2
        >>> print(OneMax(4).evaluate(np.array([True, True, False, True])))
        1
        """
        super().__init__()
        if not isinstance(n, int):
            raise TypeError(
                f"Must provide integer n, but got '{type(n)}'.")
        #: the upper bound = the length of the bit strings
        self.n: Final[int] = n
        self.evaluate = onemax  # type: ignore

    def lower_bound(self) -> int:
        """
        Get the lower bound of the onemax objective function.

        :return: 0
        :rtype: int

        >>> print(OneMax(10).lower_bound())
        0
        """
        return 0

    def upper_bound(self) -> int:
        """
        Get the upper bound of the onemax objective function.

        :return: the length of the bit string
        :rtype: int

        >>> print(OneMax(7).upper_bound())
        7
        """
        return self.n

    def __str__(self) -> str:
        """
        Get the name of the onemax objective function.

        :return: `onemax_` + length of string
        :rtype: str

        >>> print(OneMax(13))
        onemax_13
        """
        return f"onemax_{self.n}"
