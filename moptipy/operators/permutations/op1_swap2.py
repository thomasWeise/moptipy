"""
An operator swapping two elements in a permutation.

This is a unary search operator which first copies the source string `x` to
the destination string `dest`. Then it draws an index `i1` randomly.
It keeps drawing a second random index `i2` until `dest[i1] != dest[i2]`,
i.e., until the elements at the two indices are different. This will always
be true for actual permutations if `i1 != i2`, but for permutations with
repetitions, even if `i1 != i2`, sometimes `dest[i1] == dest[i2]`. Anyway,
as soon as the elements at `i1` and `i2` are different, they will be swapped.

This operator is very well-known for permutations and has been used by many
researchers, e.g., for the Traveling Salesperson Problem (TSP). In the papers
by Larrañaga et al. (1999) and Banzhaf (1990), it is called "Exchange
Mutation." It is also referred to as the "swap mutation" (Oliver et al. 1987),
point mutation operator (Ambati et al. 1991), the reciprocal exchange mutation
operator (Michalewicz 1992), or the order based mutation operator by Syswerda
(1991).

1. Pedro Larrañaga, Cindy M. H. Kuijpers, Roberto H. Murga, I. Inza, and
   S. Dizdarevic. Genetic Algorithms for the Travelling Salesman Problem: A
   Review of Representations and Operators. *Artificial Intelligence Review,*
   13(2):129-170, April 1999. Kluwer Academic Publishers, The Netherlands.
   https://doi.org/10.1023/A:1006529012972
2. Wolfgang Banzhaf. The "Molecular" Traveling Salesman. *Biological
   Cybernetics*, 64(1):7-14, November 1990, https://doi.org/10.1007/BF00203625
3. I.M. Oliver, D.J. Smith, and J.R.C. Holland. A Study of Permutation
   Crossover Operators on the Traveling Salesman Problem. In *Proceedings of
   the Second International Conference on Genetic Algorithms and their
   Application* (ICGA'87), October 1987, pages 224-230,
   https://dl.acm.org/doi/10.5555/42512.42542
4. Balamurali Krishna Ambati, Jayakrishna Ambati, and Mazen Moein Mokhtar.
   Heuristic Combinatorial Optimization by Simulated Darwinian Evolution: A
   Polynomial Time Algorithm for the Traveling Salesman Problem. *Biological
   Cybernetics,* 65(1):31-35, May 1991, https://doi.org/10.1007/BF00197287
5. Zbigniew Michalewicz. *Genetic Algorithms + Data Structures = Evolution
   Programs,* Berlin, Germany: Springer-Verlag GmbH. 1996. ISBN:3-540-58090-5.
   https://doi.org/10.1007/978-3-662-03315-9
6. Gilbert Syswerda. Schedule Optimization Using Genetic Algorithms. In
   Lawrence Davis, (ed.), *Handbook of Genetic Algorithms,* pages 332-349.
   1991. New York, NY, USA: Van Nostrand Reinhold.
7. Thomas Weise. *Optimization Algorithms.* 2021. Hefei, Anhui, China:
   Institute of Applied Optimization (IAO), School of Artificial Intelligence
   and Big Data, Hefei University. http://thomasweise.github.io/oa/

This operator performs one swap. It is similar to :class:`~moptipy.operators.\
permutations.op1_swapn.Op1SwapN`, which performs a random number of swaps.
"""
from typing import Final

import numba  # type: ignore
import numpy as np
from numpy.random import Generator

from moptipy.api.operators import Op1


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def swap_2(random: Generator, dest: np.ndarray,  # +book
           x: np.ndarray) -> None:  # +book
    """
    Copy `x` into `dest` and swap two different values in `dest`.

    :param random: the random number generator
    :param dest: the array to receive the modified copy of `x`
    :param x: the existing point in the search space

    >>> rand = np.random.default_rng(10)
    >>> xx = np.array(range(10), int)
    >>> out = np.empty(len(xx), int)
    >>> swap_2(rand, out, xx)
    >>> print(out)
    [0 1 2 3 4 5 6 9 8 7]
    >>> sum(out != xx)
    2
    >>> swap_2(rand, out, xx)
    >>> print(out)
    [0 1 7 3 4 5 6 2 8 9]
    >>> sum(out != xx)
    2
    >>> swap_2(rand, out, xx)
    >>> print(out)
    [0 1 2 3 4 8 6 7 5 9]
    >>> sum(out != xx)
    2
    """
    # start book
    dest[:] = x[:]  # First, we copy `x` to `dest`.
    length: Final[int] = len(dest)  # Get the length of `dest`.

    i1: Final[int] = random.integers(0, length)  # first random index
    v1: Final = dest[i1]  # Get the value at the first index.
    while True:  # Repeat until we find a different value.
        i2: int = random.integers(0, length)  # second random index
        v2 = dest[i2]  # Get the value at the second index.
        if v1 != v2:  # If both values different...
            dest[i2] = v1  # store v1 where v2 was
            dest[i1] = v2  # store v2 where v1 was
            return  # Exit function: we are finished.


class Op1Swap2(Op1):
    """
    A unary search operation that swaps two (different) elements.

    In other words, it performs exactly one swap on a permutation.
    It spans a neighborhood of a rather limited size but is easy
    and fast.
    """

    def __init__(self) -> None:
        """Initialize the object."""
        super().__init__()  # -book
        self.op1 = swap_2  # type: ignore  # use function directly
    # end book

    def __str__(self) -> str:
        """
        Get the name of this unary operator.

        :returns: "swap2", the name of this operator
        :retval "swap2": always
        """
        return "swap2"
