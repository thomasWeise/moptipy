"""
An operator swapping several elements in a permutation.

This operator works similarly to
:class:`~moptipy.operators.permutations.op1_swap2.Op1Swap2`, but instead of
swapping two elements (i.e., performing 1 swap), it will perform a random
number of swaps.

First, the operator copies the source string `x` into the destination `dest`.
It then chooses a random index `i1`. It remembers the element at that index
in two variables, `last` and `first`. (`last` will always be the value of the
element will next be overwritten and `first` will never change and remember
the first overwritten element, i.e., the value that we need to store again
into the array in the end.) We also initialize a variable `continue_after`
with `True`. This variable will always tell us if we need to continue and
draw another random index.

We then begin with the main loop, which is iterated as long as
`continue_after` is `True`. Directly as first action in this loop, we set
`continue_after = ri(2) <= 0`. `ri(2)` will return a random integer in
`{0, 1}`. `ri(2) <= 0` will therefore be `True` with probability 0.5.

We now will draw a new random index `i2`. We load the element at index `i2`
into variable `current`. If `current == last`, we immediately go back and
draw another random `i2`. The reason is that we want to store `current` at
`i1`, i.e., where `last` is currently stored. If `current == last`, this
would change nothing. Furthermore, if `continue_after` is `False`, then
`current` must also be different from `first`: If we exit the main loop,
then we will store `first` into the place where we found `current`. Remember,
we will overwrite the element at index `i1` with `current`, so the very first
element we overwrite must eventually be stored back into the array.

OK, if we have obtained an index `i2` whose corresponding element `current`
fulfills all requirements, we can set `dest[i1] = last = current`, i.e.,
remember `current` in `last` and also store it at index `i1`. Next, we will
overwrite the element at index `i2`, so we set `i1 = i2`. If `continue_after`
is `True`, the loop will continue. Otherwise, it will stop.

In the latter case, we store `first` back into the array at the last index
`i1` and do `dest[i1] = first`.

As a result, the operator will swap exactly 2 elements with probability 0.5.
With probability 0.25, it will swap three elements, with 0.125 probability, it
will swap 4 elements, and so on.

1. Thomas Weise. *Optimization Algorithms.* 2021. Hefei, Anhui, China:
   Institute of Applied Optimization (IAO), School of Artificial Intelligence
   and Big Data, Hefei University. http://thomasweise.github.io/oa/
"""
from typing import Callable, Final

import numpy as np
from numpy.random import Generator

from moptipy.api.operators import Op1


# start book
class Op1SwapN(Op1):
    """
    This unary search operation swaps several (different) elements.

    It is similar to `swap2`, but instead may perform a random number
    of swaps. After each swap, it decides with probability 0.5 whether
    or not to perform another swap.
    """

    def op1(self, random: Generator,
            dest: np.ndarray, x: np.ndarray) -> None:
        """
        Copy `x` into `dest` and then swap several different values.

        :param random: the random number generator
        :param dest: the array to receive the modified copy of `x`
        :param x: the existing point in the search space
        """
        np.copyto(dest, x)  # First, we copy `x` to `dest`.
        length: Final[int] = len(dest)  # Get the length of `dest`.
        ri: Final[Callable[[int], int]] = random.integers  # fast call!

        i1: int = ri(length)  # Get the first random index.
        last = first = dest[i1]  # Get the value at the first index.
        continue_after: bool = True  # True -> loop at least once.
        while continue_after:  # Repeat until we should stop
            continue_after = ri(2) <= 0  # Continue after iteration?
            while True:  # Loop forever until eligible element found.
                i2: int = ri(length)  # Get a new random index.
                current = dest[i2]  # Get the value at the new index.
                if current == last:  # If it is the same as the
                    continue  # previous value, continue.
                if continue_after or (current != first):  # If we want
                    break  # to stop, then it must be != first value.
            dest[i1] = last = current  # Store value for from i2 at i1.
            i1 = i2  # Update i1 to now point to cell of i2.
        dest[i1] = first  # Finally, store first element back at end.
    # end book

    def __str__(self) -> str:
        """
        Get the name of this unary operator.

        :returns: "swapn", the name of this operator
        :retval "swapn": always
        """
        return "swapn"
