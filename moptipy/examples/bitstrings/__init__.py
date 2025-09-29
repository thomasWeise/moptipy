"""
Example for bit-string based objective functions.

The problems noted here are mainly well-known benchmark problems from the
discrete optimization community. For these problems, we designed the dedicated
base class :class:`~moptipy.examples.bitstrings.bitstring_problem.\
BitStringProblem`.

The following benchmark problems are provided:

1.  The :mod:`~moptipy.examples.bitstrings.onemax` problem, where the
    goal is to find a bit string with the maximum number of ones.
2.  The :mod:`~moptipy.examples.bitstrings.leadingones` problem,
    where the goal is to find a bit string with the maximum number of leading
    ones.
3.  The :mod:`~moptipy.examples.bitstrings.linearharmonic` problem,
    where the goal is to find a bit string with the all ones, like in
    :mod:`~moptipy.examples.bitstrings.onemax`, but this time all bits have
    a different weight (namely their index, starting at 1).
4.  The :mod:`~moptipy.examples.bitstrings.binint` problem,
    is again similar to :mod:`~moptipy.examples.bitstrings.onemax`, but the
    bits have exponential weight. Basically, we just decode the bit string
    into an integer.
5.  The :mod:`~moptipy.examples.bitstrings.trap` problem, which is like
    OneMax, but with the optimum and worst-possible solution swapped. This
    problem is therefore highly deceptive.
6.  The :mod:`~moptipy.examples.bitstrings.twomax` problem has the global
    optimum at the string of all `1` bits and a local optimum at the string
    of all `0` bits. Both have basins of attraction of about the same size.
7.  :mod:`~moptipy.examples.bitstrings.ising1d`, the one-dimensional
    Ising model, where the goal is that all bits should have the same value as
    their neighbors in a ring.
8.  :mod:`~moptipy.examples.bitstrings.ising2d`, the two-dimensional
    Ising model, where the goal is that all bits should have the same value as
    their neighbors on a torus.
9.  The :mod:`~moptipy.examples.bitstrings.jump` problem is equivalent
    to :mod:`~moptipy.examples.bitstrings.onemax`, but has a deceptive
    region right before the optimum.
10. The :mod:`~moptipy.examples.bitstrings.plateau` problem similar to the
    :mod:`~moptipy.examples.bitstrings.jump` problem, but this time the
    optimum is surrounded by a region of neutrality.
11. The :mod:`~moptipy.examples.bitstrings.nqueens`, where the goal is to
    place `k` queens on a `k * k`-sized chess board such that no queen can
    beat any other queen.
12. The :mod:`~moptipy.examples.bitstrings.labs`, where the goal is to
    find a sequence with low autocorrelation and, thus, high merit factor.
    This problem is different from the others, because here, only for low
    values of `n`, the optimal solutions are actually known.
13. The :mod:`~moptipy.examples.bitstrings.w_model`, a benchmark
    problem with tunable epistasis, uniform neutrality, and
    ruggedness/deceptiveness.
14. The :mod:`~moptipy.examples.bitstrings.zeromax` problem, where the
    goal is to find a bit string with the maximum number of zeros. This is the
    opposite of the OneMax problem.

Parts of the code here are related to the research work of
Mr. Jiazheng ZENG (曾嘉政), a Master's student at the Institute of Applied
Optimization (应用优化研究所) of the School of
Artificial Intelligence and Big Data (人工智能与大数据学院) at
Hefei University (合肥大学) in
Hefei, Anhui, China (中国安徽省合肥市) under the supervision of
Prof. Dr. Thomas Weise (汤卫思教授).
"""
from itertools import chain
from typing import Callable, Iterable, Iterator, cast

from moptipy.examples.bitstrings.binint import BinInt
from moptipy.examples.bitstrings.bitstring_problem import BitStringProblem
from moptipy.examples.bitstrings.ising1d import Ising1d
from moptipy.examples.bitstrings.ising2d import Ising2d
from moptipy.examples.bitstrings.jump import Jump
from moptipy.examples.bitstrings.labs import LABS
from moptipy.examples.bitstrings.leadingones import LeadingOnes
from moptipy.examples.bitstrings.linearharmonic import LinearHarmonic
from moptipy.examples.bitstrings.nqueens import NQueens
from moptipy.examples.bitstrings.onemax import OneMax
from moptipy.examples.bitstrings.plateau import Plateau
from moptipy.examples.bitstrings.trap import Trap
from moptipy.examples.bitstrings.twomax import TwoMax
from moptipy.examples.bitstrings.w_model import WModel


def default_instances(
        class_scales: Callable[[
            type], Iterable[int]] = lambda _: cast("Iterable[int]", ())) \
        -> Iterator[Callable[[], BitStringProblem]]:
    """
    Get the default bit-string based benchmark instances.

    :param class_scales: a function that can override the minimum and
        maximimum problem scales on a per-benchmark-function-class
        basis. If this function returns an empty iterator, then the default
        scales are used.
    :return: an :class:`Iterable` with the default bit-string
        benchmark instances

    >>> len(list(default_instances()))
    1044

    >>> from moptipy.examples.bitstrings.binint import BinInt
    >>> from moptipy.examples.bitstrings.bitstring_problem import \
BitStringNKProblem
    >>> def clazz_scales(clazz) -> tuple[int, int]:
    ...     if issubclass(clazz, BinInt):
    ...         return 2, 30
    ...     if issubclass(clazz, BitStringNKProblem):
    ...         return 6, 16
    ...     return 8, 48
    >>> len(list(default_instances(clazz_scales)))
    331
    """
    return chain(
        BinInt.default_instances(*class_scales(BinInt)),  # type: ignore
        Ising1d.default_instances(*class_scales(Ising1d)),  # type: ignore
        Ising2d.default_instances(*class_scales(Ising2d)),  # type: ignore
        Jump.default_instances(*class_scales(Jump)),  # type: ignore
        LeadingOnes.default_instances(  # type: ignore
            *class_scales(LeadingOnes)),
        LinearHarmonic.default_instances(  # type: ignore
            *class_scales(LinearHarmonic)),
        NQueens.default_instances(*class_scales(NQueens)),  # type: ignore
        OneMax.default_instances(*class_scales(OneMax)),  # type: ignore
        Plateau.default_instances(*class_scales(Plateau)),  # type: ignore
        Trap.default_instances(*class_scales(Trap)),  # type: ignore
        TwoMax.default_instances(*class_scales(TwoMax)),  # type: ignore
        LABS.default_instances(*class_scales(LABS)),  # type: ignore
        WModel.default_instances(*class_scales(WModel)))  # type: ignore
