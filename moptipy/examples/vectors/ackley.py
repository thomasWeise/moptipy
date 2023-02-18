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
from numpy import cos, exp, mean, ndarray, sqrt

from moptipy.api.objective import Objective


@numba.njit(nogil=True, cache=True)
def ackley(x: ndarray) -> float:
    """
    Compute Ackley's function.

    :param x: the np array
    :return: the result of Ackley's function

    >>> from numpy import array
    >>> print(ackley(array([0.0, 0.0, 0.0])))
    0.0
    >>> ackley(array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    0.0
    >>> print(abs(ackley(array([0.0, 0.0, 0.0]))) < 1e-15)
    True
    >>> print(ackley(array([2.0, 2.0])))
    6.593599079287213
    >>> print(ackley(array([-3.0, 2.0])))
    7.9889108105187
    """
    # 22.718281828459045 equals 20.0 + e
    # 6.283185307179586 equals 2.0 * pi
    res = 22.718281828459045 + (-20.0 * exp(-0.2 * sqrt(
        mean(x ** 2)))) - exp(mean(cos(6.283185307179586 * x)))
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
