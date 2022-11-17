"""
Ackley's Function.

Ackley's function is a continuous, multimodal, differentiable, non-separable,
and scalable benchmark function for continuous optimization.

1. David H. Ackley. *A Connectionist Machine for Genetic Hillclimbing.* 1987.
   Volume 28 of The Kluwer International Series in Engineering and Computer
   Science. Norwell, MA, USA: Kluwer Academic Publisher.
   ISBN: 978-1-4612-9192. doi:10.1007/978-1-4613-199
"""

import numba  # type: ignore
from numpy import cos, e, exp, mean, ndarray, pi, sqrt

from moptipy.api.objective import Objective


@numba.njit(nogil=True, cache=True)
def ackley(x: ndarray) -> float:
    """
    Compute Ackley's function.

    :param x: the np array
    :return: the result of Ackley's function

    >>> from numpy import array
    >>> print(abs(ackley(array([0.0, 0.0, 0.0]))) < 1e-15)
    True
    >>> print(ackley(array([2.0, 2.0])))
    6.593599079287213
    >>> print(ackley(array([-3.0, 2.0])))
    7.9889108105187
    """
    res = 20.0 + e + (-20.0 * exp(-0.2 * sqrt(
        mean(x ** 2)))) - exp(mean(cos(2.0 * pi * x)))
    return 0.0 if res <= 0.0 else float(res)


class Ackley(Objective):
    """Ackley's function."""

    def __init__(self) -> None:
        """Initialize Ackley's function."""
        self.evaluate = ackley  # type: ignore

    def lower_bound(self) -> float:
        """
        Get the lower bound of the sphere problem.

        :return: 0

        >>> print(Ackley().lower_bound())
        0.0
        """
        return 0.0

    def __str__(self) -> str:
        """
        Get the name of Ackley's function.

        :return: `ackley`
        :retval "ackley": always

        >>> print(Ackley())
        ackley
        """
        return "ackley"
