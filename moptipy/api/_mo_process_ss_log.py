"""A multi-objective process with solution space and logging."""
from typing import Final

import numpy as np
from numpy import copyto
from pycommons.io.path import Path
from pycommons.types import type_error

from moptipy.api._mo_process_ss import _MOProcessSS
from moptipy.api._process_base import _TIME_IN_NS, _check_log_time
from moptipy.api.algorithm import Algorithm
from moptipy.api.encoding import Encoding
from moptipy.api.mo_archive import MOArchivePruner
from moptipy.api.mo_problem import MOProblem
from moptipy.api.space import Space
from moptipy.utils.logger import Logger


class _MOProcessSSLog(_MOProcessSS):
    """A multi-objective process with solution space and logging."""

    def __init__(self,
                 solution_space: Space,
                 objective: MOProblem,
                 algorithm: Algorithm,
                 pruner: MOArchivePruner,
                 archive_max_size: int,
                 archive_prune_limit: int,
                 log_file: Path | None = None,
                 search_space: Space | None = None,
                 encoding: Encoding | None = None,
                 rand_seed: int | None = None,
                 max_fes: int | None = None,
                 max_time_millis: int | None = None,
                 goal_f: int | float | None = None,
                 log_all_fes: bool = False) -> None:
        """
        Perform the internal initialization. Do not call directly.

        :param solution_space: the search- and solution space.
        :param objective: the objective function
        :param algorithm: the optimization algorithm
        :param pruner: the archive pruner
        :param archive_max_size: the maximum archive size after pruning
        :param archive_prune_limit: the archive size above which pruning will
            be performed
        :param log_file: the optional log file
        :param search_space: the search space.
        :param encoding: the encoding
        :param rand_seed: the optional random seed
        :param max_fes: the maximum permitted function evaluations
        :param max_time_millis: the maximum runtime in milliseconds
        :param goal_f: the goal objective value. if it is reached, the process
            is terminated
        :param log_all_fes: should we log all FEs?
        """
        super().__init__(solution_space=solution_space,
                         objective=objective,
                         algorithm=algorithm,
                         pruner=pruner,
                         archive_max_size=archive_max_size,
                         archive_prune_limit=archive_prune_limit,
                         log_file=log_file,
                         search_space=search_space,
                         encoding=encoding,
                         rand_seed=rand_seed,
                         max_fes=max_fes,
                         max_time_millis=max_time_millis,
                         goal_f=goal_f)
        if not isinstance(log_file, str):
            raise type_error(log_file, "log_file", str)
        if not isinstance(log_all_fes, bool):
            raise type_error(log_all_fes, "log_all_fes", bool)

        #: `True` if all FEs are logged, `False` to only log improvements.
        self.__log_all: Final[bool] = log_all_fes
        #: The in-memory log
        self.__log: list[list[int | float | np.ndarray]] = []
        #: the quick access to the log appending method
        self.__log_append = self.__log.append

    def f_evaluate(self, x, fs: np.ndarray) -> float | int:
        if self._terminated:
            if self._knows_that_terminated:
                raise ValueError("The process has been terminated and "
                                 "the algorithm knows it.")
            return self._current_best_f

        current_y: Final = self._current_y
        self._g(x, current_y)
        result: Final[int | float] = self._f_evaluate(current_y, fs)
        self._current_fes = current_fes = self._current_fes + 1
        do_term: bool = current_fes >= self._end_fes
        do_log: bool = self.__log_all
        ctn: int = 0

        improved: bool = False
        if result < self._current_best_f:
            improved = True
            self._current_best_f = result
            copyto(self._current_best_fs, fs)
            self.copy(self._current_best_x, x)
            self._current_y = self._current_best_y
            self._current_best_y = current_y
            do_term = do_term or (result <= self._end_f)

        if self.check_in(x, fs, True) or improved:
            self._last_improvement_fe = current_fes
            self._current_time_nanos = ctn = _TIME_IN_NS()
            self._last_improvement_time_nanos = ctn
            do_log = True

        if do_log:
            if ctn <= 0:
                self._current_time_nanos = ctn = _TIME_IN_NS()
            self.__log_append([current_fes, ctn, result, fs.copy()])

        if do_term:
            self.terminate()

        return result

    def _check_timing(self) -> None:
        super()._check_timing()
        _check_log_time(self._start_time_nanos, self._current_time_nanos,
                        self.__log)

    def _write_log(self, logger: Logger) -> None:
        self._write_mo_log(self.__log, self._start_time_nanos,
                           self.__log_all, logger)
        del self.__log
        super()._write_log(logger)

    def __str__(self) -> str:
        return "MOLoggingProcessWithSearchSpace"
