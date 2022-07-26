"""The base classes for multi-objective optimization algorithms."""
from typing import cast

from moptipy.api.algorithm import Algorithm
from moptipy.api.mo_process import MOProcess
from moptipy.api.process import Process
from moptipy.utils.types import type_error


class MOAlgorithm(Algorithm):
    """A base class for multi-objective optimization algorithms."""

    def solve(self, process: Process) -> None:
        """
        Forward to :meth:`solve_mo` and cast `process` to `MOProcess`.

        :param process: the process to solve. Must be an instance of
            :class:`~moptipy.api.mo_process.MOProcess`.
        """
        if not isinstance(process, MOProcess):
            raise type_error(process, "process", MOProcess)
        self.solve_mo(cast(MOProcess, process))

    def solve_mo(self, process: MOProcess) -> None:
        """
        Apply the multi-objective optimization algorithm to the given process.

        :param process: the multi-objective process which provides methods to
            access the search space, the termination criterion, and a source
            of randomness. It also wraps the objective function, remembers the
            best-so-far solution, and takes care of creating log files (if
            this is wanted).
        """
