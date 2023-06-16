"""
Perform restarts of an algorithm based on the Luby sequence.

Luby et al. showed that, for Las Vegas algorithms with known run time
distribution, there is an optimal stopping time in order to minimize the
expected running time. Even if the distribution is unknown, there is a
universal sequence of running times given by (1,1,2,1,1,2,4,1,1,2,1,1,2,4,
8,...), which is the optimal restarting strategy up to constant factors.
While this only holds for Las Vegas algorithms, this restart length may also
be used for optimization, e.g., if we aim to find a globally optimal
solution.

1. M. Luby, A. Sinclair, and S. Zuckerman. Optimal Speedup of Las Vegas
   Algorithms. *Information Processing Letters* 47(4):173-180. September 1993.
   https://doi.org/10.1016/0020-0190(93)90029-9

"""
from typing import Callable, Final, TypeVar, cast

import numba  # type: ignore

from moptipy.api.algorithm import Algorithm
from moptipy.api.mo_algorithm import MOAlgorithm
from moptipy.api.mo_process import MOProcess
from moptipy.api.process import Process
from moptipy.api.subprocesses import for_fes
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.types import check_int_range, type_error


@numba.njit(cache=True)
def _luby(i: int) -> int:
    """
    Compute the Luby sequence.

    >>> [_luby(ii) for ii in range(1, 65)]
    [1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, 1, 1, 2, 1, 1, 2, 4, 1, \
1, 2, 1, 1, 2, 4, 8, 16, 1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, \
1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, 16, 32, 1]
    """
    two_by_k: int = 1
    while True:
        two_by_k_minus_one = two_by_k
        two_by_k *= 2
        if two_by_k < two_by_k_minus_one:
            return two_by_k_minus_one
        if i == (two_by_k - 1):
            return two_by_k_minus_one
        if i >= two_by_k:
            continue
        return _luby((i - two_by_k_minus_one) + 1)


class __LubyAlgorithm(Algorithm):
    """A wrapper around an existing algorithm."""

    def __init__(self, algorithm: Algorithm, base_fes: int) -> None:
        """
        Create the algorithm wrapper.

        :param algorithm: the algorithm to wrap
        :param base_fes: the base fes
        """
        super().__init__()
        #: the algorithm
        self._algo: Final[Algorithm] = algorithm
        #: the base FEs
        self._base_fes: Final[int] = base_fes
        self.initialize = algorithm.initialize  # type: ignore # fast call

    def __str__(self) -> str:
        """
        Get the string representation of this algorithm.

        :returns: `luby_` + sub-algorithm name
        """
        return f"luby{self._base_fes}_{self._algo}"

    def solve(self, process: Process) -> None:
        """
        Solve a single-objective problem.

        :param process: the process
        """
        st: Final[Callable[[], bool]] = process.should_terminate
        rst: Final[Callable[[], None]] = self.initialize
        sv: Final[Callable[[Process], None]] = self._algo.solve
        base: Final[int] = self._base_fes
        index: int = 0
        while not st():
            index = index + 1
            with for_fes(process, base * _luby(index)) as prc:
                rst()
                sv(prc)

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Store the parameters of this luby algorithm wrapper to the logger.

        :param logger: the logger
        """
        super().log_parameters_to(logger)
        logger.key_value("baseFEs", self._base_fes)
        with logger.scope("a") as scope:
            self._algo.log_parameters_to(scope)


class __LubyMOAlgorithm(__LubyAlgorithm, MOAlgorithm):
    """A wrapper around an existing multi-objective algorithm."""

    def solve_mo(self, process: MOProcess) -> None:
        """
        Solve a multi-objective problem.

        :param process: the process
        """
        st: Final[Callable[[], bool]] = process.should_terminate
        rst: Final[Callable[[], None]] = self.initialize
        sv: Final[Callable[[MOProcess], None]] = cast(
            MOAlgorithm, self._algo).solve_mo
        base: Final[int] = self._base_fes
        index: int = 0
        while not st():
            index = index + 1
            with for_fes(process, base * _luby(index)) as prc:
                rst()
                sv(prc)

    def __str__(self) -> str:
        """
        Get the string representation of this algorithm.

        :returns: `moluby(base_fes)_` + sub-algorithm name
        """
        return f"moluby{self._base_fes}_{self._algo}"


#: the type variable for single- and multi-objective algorithms.
T = TypeVar("T", Algorithm, MOAlgorithm)


def luby_restarts(algorithm: T, base_fes: int = 64) -> T:
    """
    Perform restarts of an algorithm in the Luby fashion.

    The restart run length in FEs is determined by the Luby sequence (1, 1,
    2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, 1, 1, 2, 1, 1, 2, 4, 1, ...)
    multiplied by `base_fes`. Restarts are performed until a termination
    criterion is met.

    :param algorithm: the algorithm
    :param base_fes: the basic objective function evaluations
    """
    check_int_range(base_fes, "base_fes", 1, 1_000_000_000_000)
    if isinstance(algorithm, MOAlgorithm):
        return __LubyMOAlgorithm(algorithm, base_fes)
    if isinstance(algorithm, Algorithm):
        return __LubyAlgorithm(algorithm, base_fes)
    raise type_error(algorithm, "algorithm", (Algorithm, MOAlgorithm))
