"""
An objective function counting the number of zeros in a bit string.

This problem exists mainly for testing purposes as counterpart of
:class:`~moptipy.examples.bitstrings.onemax.OneMax`.
"""

import numba  # type: ignore
import numpy as np

from moptipy.examples.bitstrings.bitstring_problem import BitStringProblem


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


class ZeroMax(BitStringProblem):
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
        super().__init__(n)
        self.evaluate = zeromax  # type: ignore

    def __str__(self) -> str:
        """
        Get the name of the zeromax objective function.

        :return: `zeromax_` + length of string

        >>> print(ZeroMax(13))
        zeromax_13
        """
        return f"zeromax_{self.n}"
