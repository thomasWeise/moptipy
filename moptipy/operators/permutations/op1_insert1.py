"""
An operator deleting an element from a permutation and inserting it elsewhere.

The operator is similar to a combination of the `rol` and `ror` operators
in [1]. If the permutation represents, e.g., a tour in the Traveling
Salesperson Problem (TSP), then a rotation replaces two edges (if two
neighboring elements are rotated = swapped) or three edges (if at least three
elements take part in the rotation). It may therefore be considered as a
possible 3-opt move on the TSP.

An alternative way to look at this is given in [2, 3, 4], where the operator
is called "Insertion Mutation": An element of the permutation is removed from
its current position (`i1`) and inserted at another position (`i2`). The
operator is therefore called "position based mutation operator" in [5].

1. Thomas Weise, Raymond Chiong, Ke Tang, Jörg Lässig, Shigeyoshi Tsutsui,
   Wenxiang Chen, Zbigniew Michalewicz, and Xin Yao. Benchmarking Optimization
   Algorithms: An Open Source Framework for the Traveling Salesman Problem.
   *IEEE Computational Intelligence Magazine (CIM)* 9(3):40-52, August 2014.
   https://doi.org/10.1109/MCI.2014.2326101
2. Pedro Larrañaga, Cindy M. H. Kuijpers, Roberto H. Murga, I. Inza, and
   S. Dizdarevic. Genetic Algorithms for the Travelling Salesman Problem: A
   Review of Representations and Operators. *Artificial Intelligence Review,*
   13(2):129-170, April 1999. Kluwer Academic Publishers, The Netherlands.
   https://doi.org/10.1023/A:1006529012972
3. David B. Fogel. An Evolutionary Approach to the Traveling Salesman Problem.
   *Biological Cybernetics* 60(2):139-144, December 1988.
   https://doi.org/10.1007/BF00202901
4. Zbigniew Michalewicz. *Genetic Algorithms + Data Structures = Evolution
   Programs,* Berlin, Germany: Springer-Verlag GmbH. 1996. ISBN:3-540-58090-5.
   https://doi.org/10.1007/978-3-662-03315-9
5. Gilbert Syswerda. Schedule Optimization Using Genetic Algorithms. In
   Lawrence Davis, (ed.), *Handbook of Genetic Algorithms,* pages 332-349.
   1991. New York, NY, USA: Van Nostrand Reinhold.
6. Yuezhong Wu, Thomas Weise, and Raymond Chiong. Local Search for the
   Traveling Salesman Problem: A Comparative Study. In *Proceedings of the
   14th IEEE Conference on Cognitive Informatics & Cognitive Computing
   (ICCI*CC'15),* July 6-8, 2015, Beijing, China, pages 213-220, Los Alamitos,
   CA, USA: IEEE Computer Society Press, ISBN: 978-1-4673-7289-3.
   https://doi.org/10.1109/ICCI-CC.2015.7259388
7. Dan Xu, Thomas Weise, Yuezhong Wu, Jörg Lässig, and Raymond Chiong. An
   Investigation of Hybrid Tabu Search for the Traveling Salesman Problem. In
   *Proceedings of the 10th International Conference on Bio-Inspired Computing
   &mdash; Theories and Applications (BIC-TA'15),* September 25-28, 2015,
   Hefei, Anhui, China, volume 562 of Communications in Computer and
   Information Science. Berlin/Heidelberg: Springer-Verlag, pages 523-537,
   ISBN 978-3-662-49013-6. https://doi.org/10.1007/978-3-662-49014-3_47
8. Weichen Liu, Thomas Weise, Yuezhong Wu, and Raymond Chiong. Hybrid Ejection
   Chain Methods for the Traveling Salesman Problem. In *Proceedings of the
   10th International Conference on Bio-Inspired Computing &mdash; Theories
   and Applications (BIC-TA'15),* September 25-28, 2015, Hefei, Anhui, China,
   volume 562 of Communications in Computer and Information Science.
   Berlin/Heidelberg: Springer-Verlag, pages 268-282, ISBN 978-3-662-49013-6.
   https://doi.org/10.1007/978-3-662-49014-3_25
9. Yuezhong Wu, Thomas Weise, and Weichen Liu. Hybridizing Different Local
   Search Algorithms with Each Other and Evolutionary Computation: Better
   Performance on the Traveling Salesman Problem. In *Proceedings of the 18th
   Genetic and Evolutionary Computation Conference (GECCO'16),* July 20-24,
   2016, Denver, CO, USA, pages 57-58, New York, NY, USA: ACM.
   ISBN: 978-1-4503-4323-7. https://doi.org/10.1145/2908961.2909001
"""
from typing import Final

import numba  # type: ignore
import numpy as np
from numpy.random import Generator

from moptipy.api.operators import Op1


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def rotate(random: Generator, dest: np.ndarray, x: np.ndarray) -> None:
    """
    Copy `x` into `dest` and then rotate a subsequence by one step.

    The function repeatedly tries to rotate a portion of an array to the left
    or right in place. It will continue trying until something changed.
    In each step, it draws two indices `i1` and `i2`.

    If `i1 < i2`, then a left rotation by one step is performed. In other
    words, the element at index `i1 + 1` goes to index `i1`, the element at
    index `i1 + 2` goes to index `i1 + 1`, and so on. The lst element, i.e.,
    the one at index `i2` goes to index `i2 - 1`. Finally, the element that
    originally was at index `i1` goes to index `i2`. If any element in the
    array has changed, this function is done, otherwise it tries again.

    If `i1 > i2`, then a right rotation by one step is performed. In other
    words, the element at index `i1 - 1` goes to index `i1`, the element at
    index `i1 - 2` goes to index `i1 - 1`, and so on. Finally, the element
    that originally was at index `i1` goes to index `i2`. If any element in
    the array has changed, this function tries again, otherwise it stops.

    This corresponds to extracting the element at index `i1` and re-inserting
    it at index `i2`.

    :param random: the random number generator
    :param dest: the array to receive the modified copy of `x`
    :param x: the existing point in the search space

    >>> rand = np.random.default_rng(10)
    >>> xx = np.array(range(10), int)
    >>> out = np.empty(len(xx), int)
    >>> rotate(rand, out, xx)
    >>> print(out)
    [0 1 2 3 4 5 6 8 9 7]
    >>> rotate(rand, out, xx)
    >>> print(out)
    [0 1 2 3 4 5 6 8 7 9]
    >>> rotate(rand, out, xx)
    >>> print(out)
    [0 5 1 2 3 4 6 7 8 9]
    >>> rotate(rand, out, xx)
    >>> print(out)
    [0 1 2 3 4 8 5 6 7 9]
    """
    dest[:] = x[:]
    length: Final[int] = len(dest)  # Get the length of `dest`.
    unchanged: bool = True
    # try to rotate the dest array until something changes
    while unchanged:
        i1 = random.integers(0, length)
        i2 = random.integers(0, length)
        if i1 == i2:  # nothing to be done
            continue  # array will not be changed

        if i1 < i2:  # rotate to the left: move elements to lower indices?
            first = dest[i1]  # get the element to be removed
            while i1 < i2:  # iterate the indices
                i3 = i1 + 1  # get next higher index
                cpy = dest[i3]  # get next element at that higher index
                unchanged &= (cpy == dest[i1])  # is a change?
                dest[i1] = cpy  # store next element at the lower index
                i1 = i3  # move to next higher index
            unchanged &= (first == dest[i2])  # check if change
            dest[i2] = first  # store removed element at highest index
            continue

        last = dest[i1]  # last element; rotate right: move elements up
        while i2 < i1:  # iterate over indices
            i3 = i1 - 1  # get next lower index
            cpy = dest[i3]  # get element at that lower index
            unchanged &= (cpy == dest[i1])  # is a change?
            dest[i1] = cpy  # store element at higher index
            i1 = i3  # move to next lower index
        unchanged &= (last == dest[i2])  # check if change
        dest[i2] = last  # store removed element at lowest index


class Op1Insert1(Op1):
    """
    Delete an element from a permutation and insert it elsewhere.

    This operation keep drawing to random numbers, `i1` and `i2`. If
    `i1 < i2`, then the permutation is rotated one step to the left between
    `i1` and `i2`. If `i1 > i2`, then it is rotated one step to the right
    between `i1` and `i2`. If the permutation has changed, the process is
    stopped. This corresponds to extracting the element at index `i1` and
    re-inserting it at index `i2`. This operator also works for permutations
    with repetitions.
    """

    def __init__(self) -> None:
        """Initialize the object."""
        super().__init__()
        self.op1 = rotate  # type: ignore  # use function directly

    def __str__(self) -> str:
        """
        Get the name of this unary operator.

        :returns: "rot1", the name of this operator
        :retval "rot1": always
        """
        return "rot1"
