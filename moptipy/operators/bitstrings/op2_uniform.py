"""
A binary operator copying each bit from either source string.

Uniform crossover copies the value of every bit with the same probability
from either of the two parent strings. This means that all bits where both
parent strings have the same value remain unchanged and are copied directly
to the offspring. The bits where the parent strings have different values
are effectively randomized. This is easy to see from the fact that, if both
parents have different values for one bit, then one of them must have the
bit set to `1` and the other one set to `0`. This means that the value in the
offspring is set to `0` with probability `0.5` and set to `1` with probability
`0.5`. This corresponds to drawing it uniformly at random, i.e., randomizing
it.

1. Gilbert Syswerda. Uniform Crossover in Genetic Algorithms. In J. David
   Schaffer, editor, *Proceedings of the 3rd International Conference on
   Genetic Algorithms* (ICGA'89), June 4-7, 1989, Fairfax, VA, USA, pages 2-9.
   San Francisco, CA, USA: Morgan Kaufmann Publishers Inc.
   ISBN: 1-55860-066-3. https://www.researchgate.net/publication/201976488.
2. Hans-Georg Beyer and Hans-Paul Schwefel. Evolution Strategies - A
   Comprehensive Introduction. *Natural Computing: An International
   Journal* 1(1):3-52, March 2002, http://doi.org/10.1023/A:1015059928466.
   https://www.researchgate.net/publication/220132816.
3. Hans-Georg Beyer. An Alternative Explanation for the Manner in which
   Genetic Algorithms Operate. *Biosystems* 41(1):1-15, January 1997,
   https://doi.org/10.1016/S0303-2647(96)01657-7.
4. William M. Spears and Kenneth Alan De Jong. On the Virtues of Parameterized
   Uniform Crossover. In Richard K. Belew and Lashon Bernard Booker, editors,
   *Proceedings of the Fourth International Conference on Genetic Algorithms*
   (ICGA'91), July 13-16, 1991, San Diego, CA, USA, pages 230-236.
   San Francisco, CA, USA: Morgan Kaufmann Publishers Inc.
   ISBN: 1-55860-208-9. https://www.mli.gmu.edu/papers/91-95/91-18.pdf.
"""

import numba  # type: ignore
import numpy as np
from numpy.random import Generator

from moptipy.api.operators import Op2


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def uniform(random: Generator, dest: np.ndarray, x0: np.ndarray,
            x1: np.ndarray) -> None:
    """
    Perform the actual work of the uniform crossover.

    :param random: the random number generator
    :param dest: the destination array
    :param x0: the first source array
    :param x1: the second source array

    >>> a = np.full(10, True)
    >>> b = np.full(len(a), False)
    >>> r = np.random.default_rng(10)
    >>> out = np.empty(len(a), bool)
    >>> uniform(r, out, a, b)
    >>> print(out)
    [ True  True False False  True  True  True False  True  True]
    >>> uniform(r, out, a, b)
    >>> print(out)
    [False False False  True False  True False False  True  True]
    >>> uniform(r, out, a, b)
    >>> print(out)
    [False  True False False  True  True  True  True  True  True]
    >>> uniform(r, out, a, b)
    >>> print(out)
    [False  True  True False  True  True False False False  True]
    """
    for i in range(len(dest)):  # pylint: disable=C0200
        v = random.integers(0, 2)  # create boolean value
        # copy from x0 with p=0.5 and from x1 with p=0.5
        dest[i] = x1[i] if v == 0 else x0[i]


class Op2Uniform(Op2):
    """
    A binary search operation that copies each bit from either source.

    For each index `i` in the destination array `dest`, uniform
    crossover copies the value from the first source string `x0`with
    probability 0.5 and otherwise the value from the second source
    string `x1`. All bits that have the same value in `x0` and `x1`
    will retain this value in `dest`, all bits where `x0` and `x1`
    differ will effectively be randomized (be `0` with probability 0.5
    and `1` with probability 0.5).
    """

    def __init__(self):
        """Initialize the uniform crossover operator."""
        super().__init__()
        self.op2 = uniform  # type: ignore

    def __str__(self) -> str:
        """
        Get the name of this binary operator.

        :return: "uniform"
        """
        return "uniform"
