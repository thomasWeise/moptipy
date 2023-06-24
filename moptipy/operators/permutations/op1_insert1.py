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
def _rotate(arr: np.ndarray, i1: int, i2: int) -> bool:
    """
    Rotate a portion of an array to the left or right in place.

    If `i1 < i2`, then a left rotation by one step is performed. In other
    words, the element at index `i1 + 1` goes to index `i1`, the element at
    index `i1 + 2` goes to index `i1 + 1`, and so on. The lst element, i.e.,
    the one at index `i2` goes to index `i2 - 1`. Finally, the element that
    originally was at index `i1` goes to index `i2`. If any element in the
    array has changed, this function returns `False`, otherwise `True`.

    If `i1 > i2`, then a right rotation by one step is performed. In other
    words, the element at index `i1 - 1` goes to index `i1`, the element at
    index `i1 - 2` goes to index `i1 - 1`, and so on. Finally, the element
    that originally was at index `i1` goes to index `i2`. If any element in
    the array has changed, this function returns `False`, otherwise `True`.

    This corresponds to extracting the element at index `i1` and re-inserting
    it at index `i2`.

    :param arr: the array to rotate
    :param i1: the start index, in `0..len(arr)-1`
    :param i2: the end index, in `0..len(arr)-1`
    :returns: whether the array was *unchanged*
    :retval False: if the array `arr` is now different from before
    :retval True: if the array `arr` has not changed

    >>> import numpy as npx
    >>> dest = npx.array(range(10))
    >>> print(dest)
    [0 1 2 3 4 5 6 7 8 9]
    >>> _rotate(dest, 3, 4)
    False
    >>> print(dest)
    [0 1 2 4 3 5 6 7 8 9]
    >>> _rotate(dest, 3, 4)
    False
    >>> print(dest)
    [0 1 2 3 4 5 6 7 8 9]
    >>> _rotate(dest, 4, 3)
    False
    >>> print(dest)
    [0 1 2 4 3 5 6 7 8 9]
    >>> _rotate(dest, 4, 3)
    False
    >>> print(dest)
    [0 1 2 3 4 5 6 7 8 9]
    >>> _rotate(dest, 3, 6)
    False
    >>> print(dest)
    [0 1 2 4 5 6 3 7 8 9]
    >>> _rotate(dest, 6, 3)
    False
    >>> print(dest)
    [0 1 2 3 4 5 6 7 8 9]
    >>> _rotate(dest, 0, len(dest) - 1)
    False
    >>> print(dest)
    [1 2 3 4 5 6 7 8 9 0]
    >>> _rotate(dest, len(dest) - 1, 0)
    False
    >>> print(dest)
    [0 1 2 3 4 5 6 7 8 9]
    >>> _rotate(dest, 7, 7)
    True
    >>> dest = np.array([0, 1, 2, 3, 3, 3, 3, 3, 8, 9])
    >>> _rotate(dest, 7, 7)
    True
    >>> _rotate(dest, 4, 6)
    True
    >>> print(dest)
    [0 1 2 3 3 3 3 3 8 9]
    >>> _rotate(dest, 6, 4)
    True
    >>> print(dest)
    [0 1 2 3 3 3 3 3 8 9]
    >>> _rotate(dest, 4, 7)
    True
    >>> print(dest)
    [0 1 2 3 3 3 3 3 8 9]
    >>> _rotate(dest, 6, 7)
    True
    >>> print(dest)
    [0 1 2 3 3 3 3 3 8 9]
    >>> _rotate(dest, 4, 8)
    False
    >>> print(dest)
    [0 1 2 3 3 3 3 8 3 9]
    >>> _rotate(dest, 8, 4)
    False
    >>> print(dest)
    [0 1 2 3 3 3 3 3 8 9]
    >>> _rotate(dest, 9, 4)
    False
    >>> print(dest)
    [0 1 2 3 9 3 3 3 3 8]
    >>> _rotate(dest, 4, 9)
    False
    >>> print(dest)
    [0 1 2 3 3 3 3 3 8 9]
    """
    if i1 == i2:  # nothing to be done
        return True  # array will not be changed

    unchanged: bool = True  # was the array unchanged?

    if i1 < i2:  # rotate left
        first = arr[i1]
        while i1 < i2:
            i3 = i1 + 1
            cpy = arr[i3]
            unchanged = unchanged and (cpy == arr[i1])
            arr[i1] = cpy
            i1 = i3
        unchanged = unchanged and (first == arr[i2])
        arr[i2] = first
        return unchanged

    last = arr[i1]
    while i2 < i1:  # rotate right
        i3 = i1 - 1
        cpy = arr[i3]
        unchanged = unchanged and (cpy == arr[i1])
        arr[i1] = cpy
        i1 = i3
    unchanged = unchanged and (last == arr[i2])
    arr[i2] = last
    return unchanged


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def rotate(random: Generator, dest: np.ndarray, x: np.ndarray) -> None:
    """
    Copy `x` into `dest` and then rotate a subsequence by one step.

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

    # try to rotate the dest array until something changes
    while _rotate(dest, random.integers(0, length),
                  random.integers(0, length)):
        pass


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
