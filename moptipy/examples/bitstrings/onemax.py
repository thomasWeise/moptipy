"""An objective function counting the number of ones in a bit string."""
from typing import Final

import numba  # type: ignore
import numpy as np

from moptipy.api.objective import Objective
from moptipy.spaces.bitstrings import BitStrings


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
    return len(x) - x.sum()


class OneMax(Objective):
    """This objective function returns the number of ones in a bit string."""

    def __init__(self, space: BitStrings) -> None:  # +book
        """
        Initialize the onemax objective function.

        :param BitStrings space: the bit string space
        """
        super().__init__()
        if not isinstance(space, BitStrings):
            raise TypeError(
                f"Must provide BitStrings, but got '{type(space)}'.")
        #: the upper bound = the length of the bit strings
        self.__ub: Final[int] = space.dimension
        self.evaluate = onemax  # type: ignore

    def lower_bound(self) -> int:
        """
        Get the lower bound of the onemax objective function.

        :return: 0
        :rtype: int
        """
        return 0

    def upper_bound(self) -> int:
        """
        Get the upper bound of the onemax objective function.

        :return: the length of the bit string
        :rtype: int
        """
        return self.__ub

    def get_name(self) -> str:
        """
        Get the name of the onemax objective function.

        :return: `onemax_` + length of string
        :rtype: str
        """
        return f"onemax_{self.__ub}"
