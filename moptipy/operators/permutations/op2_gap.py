"""
A binary operator that merges two permutations by adding elements in sequence.

Assume we have two permutations `x0` and `x1`. For each position `i` in the
destination string `dest`, we randomly select the permutation `x` from which
the value should come (so either `x=x0` or `x=x1`). We then store the first
value not yet marked as done from `x` in `dest[i]`, mark that value as
done both in `x0` and `x1`.

This operator may be considered as a generalized version of the Alternating
Position Crossover operator (AP) for the Traveling Salesperson Problem by
Larrañaga et al. (1997). The original AP operator, as described by Larrañaga
et al., simply creates an offspring by selecting alternately the next element
of the fist parent and the next element of the second parent, omitting the
elements already present in the offspring. For example, if `x0` is `12345678`
and `x1` is `37516824`, the AP operator gives the following offspring
`13275468`. Exchanging the parents results in `31725468`.

Our generalized version randomly decides which of the two parent permutations
to use each time, hopefully resulting in a greater variety of possible results.
It also does not skip over a parent if its next element is already used, but
instead picks the next not-yet-used element from that parent.
As mnemonic for the operator, we use `gap`. Larrañaga et al. used `AP` for the
version of the operator that strictly alternates between parents.

1. Pedro Larrañaga, Cindy M. H. Kuijpers, Mikel Poza, and Roberto H. Murga.
   Decomposing Bayesian Networks: Triangulation of the Moral Graph with Genetic
   Algorithms, *Statistics and Computing,* 7(1):19-34, March 1997,
   https://doi.org/10.1023/A:1018553211613
2. Pedro Larrañaga, Cindy M. H. Kuijpers, Roberto H. Murga, Iñaki Inza, and
   S. Dizdarevic. Genetic Algorithms for the Travelling Salesman Problem: A
   Review of Representations and Operators. *Artificial Intelligence Review,*
   13(2):129-170, April 1999. Kluwer Academic Publishers, The Netherlands.
   https://doi.org/10.1023/A:1006529012972
"""
from typing import Final

import numba  # type: ignore
import numpy as np
from numpy.random import Generator

from moptipy.api.operators import Op2
from moptipy.spaces.permutations import Permutations
from moptipy.utils.nputils import DEFAULT_BOOL
from moptipy.utils.types import type_error


@numba.njit(nogil=True, cache=True)
# start book
def _op2_gap(r: np.ndarray, dest: np.ndarray,
             x0: np.ndarray, x1: np.ndarray,
             x0_done: np.ndarray, x1_done: np.ndarray) -> None:
    """
    Apply a sequence mix from `x0` and `x1` to `dest`.

    :param r: the random numbers, of length `n - 1` (!!)
    :param dest: the array to receive the result
    :param x0: the first existing point in the search space
    :param x1: the second existing point in the search space
    :param x0_done: a boolean array marking the elements in `x0` that have
        been used
    :param x1_done: a boolean array marking the elements in `x1` that have
        been used
    """
    x0_done.fill(False)  # all values in x0 are available
    x1_done.fill(False)  # all values in x1 are available
    length: Final[int] = len(x0)

    desti: int = 0  # writing to dest starts at index 0
    x0i: int = 0  # first valid value in x0 is at index 0
    x1i: int = 0  # first valid value in x1 is at index 0
    for rr in r:  # repeat until dest is filled, i.e., desti=length
        # randomly chose a source point and pick next operation
        value: int = x0[x0i] if rr == 0 else x1[x1i]
        dest[desti] = value  # store the value in the destination
        desti = desti + 1  # step destination index

        for x0j in range(x0i, length):  # mark value as done in x0
            if (x0[x0j] == value) and (not x0_done[x0j]):  # find
                x0_done[x0j] = True  # value is found and not done
                break  # so we mark it as done and break the loop
        while x0_done[x0i]:  # now we find the next not-yet-done
            x0i = x0i + 1  # value in x0

        for x1j in range(x1i, length):  # mark value as done in x1
            if (x1[x1j] == value) and (not x1_done[x1j]):  # find
                x1_done[x1j] = True  # value is found and not done
                break  # so we mark it as done and break the loop
        while x1_done[x1i]:  # now we find the next not-yet-done
            x1i = x1i + 1  # value in x1

    dest[desti] = x0[x0i]  # = x1[x1i]: the final missing value
# end book


# start book
class Op2GeneralizedAlternatingPosition(Op2):
    """A binary operator trying to preserve the value sequence."""

    def op2(self, random: Generator, dest: np.ndarray,
            x0: np.ndarray, x1: np.ndarray) -> None:
        """
        Apply a sequence mix from `x0` and `x1` to `dest`.

        :param random: the random number generator
        :param dest: the array to receive the result
        :param x0: the first existing point in the search space
        :param x1: the second existing point in the search space
        """
        _op2_gap(random.integers(low=2, high=None, size=len(dest) - 1),
                 dest, x0, x1, self.__x0_done, self.__x1_done)
    # end book

    def __init__(self, space: Permutations) -> None:
        """
        Initialize the GAP crossover operator.

        :param space: the permutation space
        """
        super().__init__()
        if not isinstance(space, Permutations):
            raise type_error(space, "space", Permutations)
        #: the elements that are done in `x0`
        self.__x0_done: Final[np.ndarray] = np.ndarray(
            (space.dimension,), DEFAULT_BOOL)
        #: the elements that are done in `x1`
        self.__x1_done: Final[np.ndarray] = np.ndarray(
            (space.dimension,), DEFAULT_BOOL)

    def __str__(self) -> str:
        """
        Get the name of this binary operator.

        :returns: "gap" for "generalized alternating position crossover",
            the name of this operator
        :retval "gap": always
        """
        return "gap"
