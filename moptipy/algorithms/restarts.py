"""Perform restarts of an algorithm that terminates too early."""
from typing import Callable, Final, TypeVar, cast

from moptipy.api.algorithm import Algorithm
from moptipy.api.mo_algorithm import MOAlgorithm
from moptipy.api.mo_process import MOProcess
from moptipy.api.process import Process
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.types import type_error


class __RestartAlgorithm(Algorithm):
    """A wrapper around an existing algorithm."""

    def __init__(self, algorithm: Algorithm) -> None:
        """
        Create the algorithm wrapper.

        :param algorithm: the algorithm to wrap
        """
        super().__init__()
        #: the algorithm
        self._algo: Final[Algorithm] = algorithm
        self.initialize = algorithm.initialize  # type: ignore # fast call

    def __str__(self) -> str:
        """
        Get the string representation of this algorithm.

        :returns: `rs_` + sub-algorithm name
        """
        return f"rs_{self._algo}"

    def solve(self, process: Process) -> None:
        """
        Solve a single-objective problem.

        :param process: the process
        """
        st: Final[Callable[[], bool]] = process.should_terminate
        rst: Final[Callable[[], None]] = self._algo.initialize
        sv: Final[Callable[[Process], None]] = self._algo.solve
        while not st():
            rst()
            sv(process)

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Store the parameters of this algorithm wrapper to the logger.

        :param logger: the logger
        """
        super().log_parameters_to(logger)
        with logger.scope("a") as scope:
            self._algo.log_parameters_to(scope)


class __RestartMOAlgorithm(__RestartAlgorithm, MOAlgorithm):
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
        while not st():
            rst()
            sv(process)

    def __str__(self) -> str:
        """
        Get the string representation of this algorithm.

        :returns: `mors_` + sub-algorithm name
        """
        return f"mors_{self._algo}"


#: the type variable for single- and multi-objective algorithms.
T = TypeVar("T", Algorithm, MOAlgorithm)


def restarts(algorithm: T) -> T:
    """
    Perform restarts of an algorithm until the termination criterion is met.

    :param algorithm: the algorithm
    """
    if isinstance(algorithm, MOAlgorithm):
        return __RestartMOAlgorithm(algorithm)
    if isinstance(algorithm, Algorithm):
        return __RestartAlgorithm(algorithm)
    raise type_error(algorithm, "algorithm", (Algorithm, MOAlgorithm))
