"""
An operator trying to swap a given number of elements in a permutation.

This operator is similar to
:mod:`~moptipy.operators.permutations.op1_swap_exactly_n`
in that it tries to swap a fixed number of elements in a permutation. It
implements the :class:`~moptipy.api.operators.Op1WithStepSize` interface.
Different from :mod:`~moptipy.operators.permutations.op1_swap_exactly_n`,
however, it does so less strictly. It applies a simple best-effort approach
and if that does not work out, then so be it. It is therefore faster, but
adheres less strictly to the give `step_size`.

This operator will always swap the right number of elements on normal
permutations. On permutations with repetitions, it enforces the number of
swaps less strongly compared to
:mod:`~moptipy.operators.permutations.op1_swap_exactly_n`, but it will be
faster either way.
"""
from typing import Final

import numba  # type: ignore
import numpy as np
from numpy.random import Generator

from moptipy.api.operators import Op1WithStepSize
from moptipy.operators.tools import exponential_step_size
from moptipy.spaces.permutations import Permutations
from moptipy.utils.nputils import DEFAULT_INT, fill_in_canonical_permutation
from moptipy.utils.types import type_error


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def swap_try_n(random: Generator, dest: np.ndarray, x: np.ndarray,
               step_size: float, indices: np.ndarray) -> None:
    """
    Copy `x` into `dest` and then swap several different values.

    :param random: the random number generator
    :param dest: the array to receive the modified copy of `x`
    :param x: the existing point in the search space
    :param step_size: the number of elements to swap
    :param indices: the array with indices

    >>> xx = np.array(range(10), int)
    >>> out = np.empty(len(xx), xx.dtype)
    >>> idxs = np.array(range(len(xx)), int)
    >>> gen = np.random.default_rng(10)
    >>> swap_try_n(gen, out, xx, 0.1, idxs)
    >>> print(out)
    [0 1 2 3 4 5 6 8 7 9]
    >>> swap_try_n(gen, out, xx, 0.1, idxs)
    >>> print(out)
    [0 2 1 3 4 5 6 7 8 9]
    >>> swap_try_n(gen, out, xx, 1.0, idxs)
    >>> print(out)
    [3 7 4 5 8 6 9 0 1 2]
    """
    dest[:] = x[:]  # First, we copy `x` to `dest`.
    remaining: int = len(dest)  # Get the length of `dest`.
# Compute the real step size based on the length of the permutation.
    steps: int = exponential_step_size(step_size, 2, remaining)

    ii: int = random.integers(0, remaining)  # Select first random index.
    i1 = indices[ii]  # Get the actual index.
    remaining = remaining - 1  # There is one less remaining index.
    indices[remaining], indices[ii] = i1, indices[remaining]

    last = first = dest[i1]  # Get the value at the first index.
    continue_after: bool = True  # True -> loop at least once.
    while continue_after:  # Repeat until we should stop
        steps = steps - 1
        continue_after = (steps > 1) and (remaining > 1)
        tested: int = remaining  # the indices that we can test.
        while True:  # Loop forever until eligible element found.
            ii = random.integers(0, tested)  # Get a new random index.
            i2 = indices[ii]  # Get the actual index.
            current = dest[i2]  # Get the value at the new index.
            if tested <= 1:  # Are all remaining elements same?
                continue_after = False  # If yes, then we quit.
                break
            if (current != last) and (
                    continue_after or (current != first)):
                remaining = remaining - 1
                indices[remaining], indices[ii] = \
                    i2, indices[remaining]
                break  # to stop, then it must be != first value.
            tested = tested - 1  # Now there is one fewer index.
            indices[tested], indices[ii] = i2, indices[tested]
        dest[i1] = last = current  # Store value for from i2 at i1.
        i1 = i2  # Update i1 to now point to cell of i2.
    dest[i1] = first  # Finally, store first element back at end.


class Op1SwapTryN(Op1WithStepSize):
    """An operator trying to swap a given number of elements."""

    def op1(self, random: Generator, dest: np.ndarray, x: np.ndarray,
            step_size: float = 0.0) -> None:
        """
        Copy `x` into `dest` and then swap several different values.

        :param random: the random number generator
        :param dest: the array to receive the modified copy of `x`
        :param x: the existing point in the search space
        :param step_size: the number of elements to swap
        """
        swap_try_n(random, dest, x, step_size, self.__indices)

    def __init__(self, perm: Permutations) -> None:
        """
        Initialize the operator.

        :param perm: the base permutation
        """
        super().__init__()
        if not isinstance(perm, Permutations):
            raise type_error(perm, "perm", Permutations)
        #: the valid indices
        self.__indices: Final[np.ndarray] = np.empty(
            perm.dimension, DEFAULT_INT)

    def initialize(self) -> None:
        """Initialize this operator."""
        super().initialize()
        fill_in_canonical_permutation(self.__indices)

    def __str__(self) -> str:
        """
        Get the name of this unary operator.

        :returns: "swaptn", the name of this operator
        :retval "swaptn": always
        """
        return "swaptn"
