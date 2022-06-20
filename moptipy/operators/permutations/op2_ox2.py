"""
The Order-based Crossover operator.

Larrañaga et al. describe this operator as follows:

The order-based crossover operator by Syswerda (1991) selects at random
several positions in a parent tour, and the order of the cities in the
selected positions of this parent is imposed on the other parent. For
example, consider the parents `12345678` and `24687531`. Suppose that in the
second parent the second, third, and sixth positions are selected. The values
in these positions are `4`, `6`, and `5` respectively. In the first parent
these cities are present at the fourth, fifth and sixth positions.
Now the offspring is equal to parent 1 except in the fourth, fifth and sixth
positions: `123xxx78`. We add the missing cities to the offspring in the same
order in which they appear in the second parent tour. This results in
`12346578`. Exchanging the role of the first parent and the second parent
gives, using the same selected positions, `24387561`.

We implement this operator such that each position has the same chance to be
chosen by either parents, i.e., the total number of positions copied from
the parents is binomially distributed with `p=0.5`, but we ensure that at
least two positions are copied from either parents (as the result would
otherwise necessarily equal one of the parents). We also switch the role of
the two parents in our implementation.

As mnemonic for the operator, we use `ox2`, similar to Larrañaga et al., who
used `OX2`.

1. G. Syswerda. Schedule Optimization Using Genetic Algorithms. In Lawrence
   Davis, L. (ed.), *Handbook of Genetic Algorithms,* pages 332–349.
   New York: Van Nostrand Reinhold.
2. Pedro Larrañaga, Cindy M. H. Kuijpers, Roberto H. Murga, Iñaki Inza, and
   S. Dizdarevic. Genetic Algorithms for the Travelling Salesman Problem: A
   Review of Representations and Operators. *Artificial Intelligence Review,*
   13(2):129–170, April 1999. Kluwer Academic Publishers, The Netherlands.
   https://doi.org/10.1023/A:1006529012972
"""
from typing import Final, Callable

import numpy as np
from numpy.random import Generator

from moptipy.api.operators import Op2
from moptipy.spaces.permutations import Permutations
from moptipy.utils.nputils import DEFAULT_BOOL, DEFAULT_INT
from moptipy.utils.types import type_error


# start book
class Op2OrderBased(Op2):
    """The order-based crossover operator."""

    def op2(self, random: Generator, dest: np.ndarray,
            x0: np.ndarray, x1: np.ndarray) -> None:
        """
        Apply the order-based crossover from `x0` and `x1` to `dest`.

        :param random: the random number generator
        :param dest: the array to receive the result
        :param x0: the first existing point in the search space
        :param x1: the second existing point in the search space
        """
        # end book
        indices: Final[np.ndarray] = self.__indices
        x1_done: Final[np.ndarray] = self.__x1_done
        x1_done.fill(False)  # all values in x1 are available
        ri: Final[Callable[[int], int]] = random.integers
        rbin: Final[Callable[[int, float], int]] = random.binomial
        length: Final[int] = len(indices)  # get length of string
        copy_from_x0: int  # the end index of copying from x0
        value: int  # the current value to be written to dest

        # start book
        while True:  # sample the number of values to copy from x0
            copy_from_x0 = rbin(length, 0.5)  # p=0.5 for each value
            if 1 < copy_from_x0 < (length - 1):  # ensure difference by
                break  # copying at least two values from each parent
        copy_from_x0 = length - copy_from_x0  # compute end index-index

        i: int = length  # the index into indices we iterate over
        mode: bool = True  # mode: True = copy from x0, False = from x1
        x1i: int = 0  # the index of the next unused value from x1
        while True:  # loop until we are finished
            index_i: int = ri(i)  # pick a random index-index
            index: int = indices[index_i]  # load the actual index
            i = i - 1  # reduce the number of values
            indices[i], indices[index_i] = index, indices[i]  # swap

            if mode:  # copy from x0 to dest
                dest[index] = value = x0[index]  # get and store value
                for x1j in range(x1i, length):  # mark as used
                    if (x1[x1j] == value) and (not x1_done[x1j]):
                        x1_done[x1j] = True  # mark value as used
                        break  # exit inner loop
                if copy_from_x0 == i:  # are we done with copying?
                    mode = False  # set mode to load from x1
                    x1i = 0  # reset x1 index
            else:  # go to next iteration
                dest[index] = x1[x1i]  # and store it in dest
                if i == 0:  # check if we are done
                    return  # ok, we are finished
                x1i = x1i + 1  # and move on to the next value
            while x1_done[x1i]:  # step x1i to next unused value
                x1i = x1i + 1  # increment
    # end book

    def __init__(self, space: Permutations) -> None:
        """
        Initialize the sequence crossover operator.

        :param space: the permutation space
        """
        super().__init__()
        if not isinstance(space, Permutations):
            raise type_error(space, "space", Permutations)
        if space.dimension < 4:
            raise ValueError(
                f"dimension must be > 3, but got {space.dimension}.")
        #: the valid indices
        self.__indices: Final[np.ndarray] = np.array(
            range(space.dimension), DEFAULT_INT)
        #: the elements that are done in `x1`
        self.__x1_done: Final[np.ndarray] = np.ndarray(
            (space.dimension, ), DEFAULT_BOOL)

    def __str__(self) -> str:
        """
        Get the name of this binary operator.

        :returns: "ox2", for "order-based crossover", the name of this
            operator
        :retval "ox2": always
        """
        return "ox2"
