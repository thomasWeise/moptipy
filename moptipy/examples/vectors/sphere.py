"""The welll-known sphere function."""

import numba  # type: ignore
import numpy as np

from moptipy.api.objective import Objective


@numba.njit(nogil=True, cache=True)
def sphere(x: np.ndarray) -> float:
    """
    Get the sum of the squares of the elements in an array.

    :param x: the np array
    :return: the number of ones

    >>> print(sphere(np.array([1.0, -2.0, 3.0])))
    14.0
    >>> print(sphere(np.array([0.0, 0.0, 0.0])))
    0.0
    """
    return float(np.sum(x ** 2))


class Sphere(Objective):
    """The well-known sphere function."""

    def __init__(self) -> None:
        """Initialize the sphere function."""
        self.evaluate = sphere  # type: ignore

    def lower_bound(self) -> float:
        """
        Get the lower bound of the sphere problem.

        :return: 0

        >>> print(Sphere().lower_bound())
        0.0
        """
        return 0.0

    def __str__(self) -> str:
        """
        Get the name of the sphere problem.

        :return: `sphere`

        >>> print(Sphere())
        sphere
        """
        return "sphere"
