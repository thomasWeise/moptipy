"""A multi-objective process with different search and solution spaces."""

from math import isfinite
from typing import Callable, Final

import numpy as np
from numpy import copyto

from moptipy.api._mo_process_no_ss import _MOProcessNoSS
from moptipy.api._process_base import _TIME_IN_NS
from moptipy.api.algorithm import Algorithm
from moptipy.api.encoding import Encoding, check_encoding
from moptipy.api.logging import (
    PREFIX_SECTION_ARCHIVE,
    SCOPE_ENCODING,
    SCOPE_SEARCH_SPACE,
    SECTION_RESULT_X,
    SUFFIX_SECTION_ARCHIVE_X,
    SUFFIX_SECTION_ARCHIVE_Y,
)
from moptipy.api.mo_archive import MOArchivePruner, MORecord
from moptipy.api.mo_problem import MOProblem
from moptipy.api.space import Space, check_space
from moptipy.utils.logger import KeyValueLogSection, Logger
from moptipy.utils.path import Path
from moptipy.utils.types import type_error


class _MOProcessSS(_MOProcessNoSS):
    """A class implementing a process with search and solution space."""

    def __init__(self,
                 solution_space: Space,
                 objective: MOProblem,
                 algorithm: Algorithm,
                 pruner: MOArchivePruner,
                 archive_max_size: int,
                 archive_prune_limit: int,
                 log_file: Path | None = None,
                 search_space: Space = None,
                 encoding: Encoding = None,
                 rand_seed: int | None = None,
                 max_fes: int | None = None,
                 max_time_millis: int | None = None,
                 goal_f: int | float | None = None) -> None:
        """
        Perform the internal initialization. Do not call directly.

        :param solution_space: the solution space.
        :param objective: the objective function
        :param algorithm: the optimization algorithm
        :param pruner: the archive pruner
        :param archive_max_size: the maximum archive size after pruning
        :param archive_prune_limit: the archive size above which pruning will
            be performed
        :param search_space: the search space.
        :param encoding: the encoding
        :param log_file: the optional log file
        :param rand_seed: the optional random seed
        :param max_fes: the maximum permitted function evaluations
        :param max_time_millis: the maximum runtime in milliseconds
        :param goal_f: the goal objective value. if it is reached, the
            process is terminated
        """
        super().__init__(solution_space=solution_space,
                         objective=objective,
                         algorithm=algorithm,
                         pruner=pruner,
                         archive_max_size=archive_max_size,
                         archive_prune_limit=archive_prune_limit,
                         log_file=log_file,
                         rand_seed=rand_seed,
                         max_fes=max_fes,
                         max_time_millis=max_time_millis,
                         goal_f=goal_f)

        #: The search space.
        self._search_space: Final[Space] = check_space(search_space)
        #: The encoding.
        self._encoding: Final[Encoding] = check_encoding(encoding)
        #: the internal encoder
        self._g: Final[Callable] = encoding.decode
        #: The holder for the currently de-coded solution.
        self._current_y = solution_space.create()
        #: The current best point in the search space.
        self._current_best_x: Final = search_space.create()
        # wrappers
        self.create = search_space.create  # type: ignore
        self.copy = search_space.copy  # type: ignore
        self.to_str = search_space.to_str  # type: ignore
        self.is_equal = search_space.is_equal  # type: ignore
        self.from_str = search_space.from_str  # type: ignore
        self.n_points = search_space.n_points  # type: ignore
        self.validate = search_space.validate  # type: ignore
        self._create_y = solution_space.create  # the y creator

    def f_evaluate(self, x, fs: np.ndarray) -> float | int:
        if self._terminated:
            if self._knows_that_terminated:
                raise ValueError("The process has been terminated and the "
                                 "algorithm knows it.")
            return self._current_best_f

        current_y: Final = self._current_y
        self._g(x, current_y)
        result: Final[int | float] = self._f_evaluate(current_y, fs)
        self._current_fes = current_fes = self._current_fes + 1
        do_term: bool = current_fes >= self._end_fes

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

        if do_term:
            self.terminate()

        return result

    def get_copy_of_best_x(self, x) -> None:
        if self._current_fes > 0:
            return self.copy(x, self._current_best_x)
        raise ValueError("No current best x available.")

    def get_copy_of_best_y(self, y) -> None:
        if self._current_fes > 0:
            return self._copy_y(y, self._current_best_y)
        raise ValueError("No current best y available.")

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        super().log_parameters_to(logger)
        with logger.scope(SCOPE_SEARCH_SPACE) as sc:
            self._search_space.log_parameters_to(sc)
        with logger.scope(SCOPE_ENCODING) as sc:
            self._encoding.log_parameters_to(sc)

    def _write_result(self, logger: Logger) -> None:
        with logger.text(SECTION_RESULT_X) as txt:
            txt.write(self._search_space.to_str(self._current_best_x))
        super()._write_result(logger)

    def _validate_x(self) -> None:
        """Validate x, if it exists."""
        self._search_space.validate(self._current_best_x)

    def _log_and_check_archive_entry(self, index: int, rec: MORecord,
                                     logger: Logger) -> int | float:
        """
        Write an archive entry.

        :param index: the index of the entry
        :param rec: the record to verify
        :param logger: the logger
        :returns: the objective value
        """
        self.validate(rec.x)
        self.f_validate(rec.fs)
        tfs: Final[np.ndarray] = self._fs_temp

        current_y: Final = self._current_y
        self._g(rec.x, current_y)
        self._solution_space.validate(current_y)
        f: Final[int | float] = self._f_evaluate(current_y, tfs)

        if not np.array_equal(tfs, rec.fs):
            raise ValueError(
                f"expected {rec.fs} but got {tfs} when re-evaluating {rec}")
        if not isinstance(f, (int, float)):
            raise type_error(f, "scalarized objective value", (int, float))
        if not isfinite(f):
            raise ValueError(f"scalarized objective value {f} is not finite")

        with logger.text(f"{PREFIX_SECTION_ARCHIVE}{index}"
                         f"{SUFFIX_SECTION_ARCHIVE_X}") as lg:
            lg.write(self.to_str(rec.x))

        with logger.text(f"{PREFIX_SECTION_ARCHIVE}{index}"
                         f"{SUFFIX_SECTION_ARCHIVE_Y}") as lg:
            lg.write(self._solution_space.to_str(current_y))

        return f

    def __str__(self) -> str:
        return "MOProcessWithSearchSpace"
