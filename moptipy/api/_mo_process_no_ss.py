"""Providing a multi-objective process without logging with a single space."""

from math import isfinite
from typing import Any, Callable, Final, cast

import numpy as np
from numpy import copyto

from moptipy.api._process_base import _TIME_IN_NS, _ns_to_ms, _ProcessBase
from moptipy.api.algorithm import Algorithm
from moptipy.api.logging import (
    KEY_ARCHIVE_F,
    KEY_ARCHIVE_MAX_SIZE,
    KEY_ARCHIVE_PRUNE_LIMIT,
    KEY_ARCHIVE_SIZE,
    KEY_BEST_FS,
    PREFIX_SECTION_ARCHIVE,
    PROGRESS_CURRENT_F,
    PROGRESS_FES,
    PROGRESS_TIME_MILLIS,
    SCOPE_PRUNER,
    SECTION_ARCHIVE_QUALITY,
    SECTION_PROGRESS,
    SUFFIX_SECTION_ARCHIVE_Y,
)
from moptipy.api.mo_archive import (
    MOArchivePruner,
    MORecord,
    check_mo_archive_pruner,
)
from moptipy.api.mo_problem import MOProblem
from moptipy.api.mo_process import MOProcess
from moptipy.api.space import Space
from moptipy.utils.logger import KeyValueLogSection, Logger
from moptipy.utils.nputils import array_to_str, np_to_py_number
from moptipy.utils.path import Path
from moptipy.utils.types import type_error


class _MOProcessNoSS(MOProcess, _ProcessBase):
    """
    An internal class process implementation.

    This class implements a stand-alone process without explicit logging where
    the search and solution space are the same.
    """

    def __init__(self,
                 solution_space: Space,
                 objective: MOProblem,
                 algorithm: Algorithm,
                 pruner: MOArchivePruner,
                 archive_max_size: int,
                 archive_prune_limit: int,
                 log_file: Path | None = None,
                 rand_seed: int | None = None,
                 max_fes: int | None = None,
                 max_time_millis: int | None = None,
                 goal_f: int | float | None = None) -> None:
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
        :param rand_seed: the optional random seed
        :param max_fes: the maximum permitted function evaluations
        :param max_time_millis: the maximum runtime in milliseconds
        :param goal_f: the goal objective value. if it is reached, the process
            is terminated
        """
        _ProcessBase.__init__(
            self, solution_space, objective, algorithm, log_file, rand_seed,
            max_fes, max_time_millis, goal_f)
        self.f_dimension = objective.f_dimension  # type: ignore
        self.f_create = objective.f_create  # type: ignore
        self.f_validate = objective.f_validate  # type: ignore
        self.f_dtype = objective.f_dtype  # type: ignore
        self.f_dominates = objective.f_dominates  # type: ignore

        #: the internal evaluation function
        self._f_evaluate: Final[Callable[
            [Any, np.ndarray], int | float]] = objective.f_evaluate

        #: the temporary variable for objective function evaluations
        self._fs_temp: Final[np.ndarray] = self.f_create()
        #: the holder for the objective vector of the current best solution
        self._current_best_fs: Final[np.ndarray] = self.f_create()
        #: the internal archive pruner
        self._pruner: Final[MOArchivePruner] = check_mo_archive_pruner(pruner)
        #: the fast call to the pruning routine
        self._prune: Final[Callable[[list[MORecord], int, int], None]] \
            = pruner.prune
        if not isinstance(archive_max_size, int):
            raise TypeError(archive_max_size, "archive_max_size", int)
        if not isinstance(archive_prune_limit, int):
            raise TypeError(archive_prune_limit, "archive_prune_limit", int)
        if not (0 < archive_max_size <= archive_prune_limit):
            raise ValueError(
                f"invalid archive_max_size={archive_max_size} and "
                f"archive_prune_limit={archive_prune_limit} combination")
        #: the maximum archive size
        self._archive_max_size: Final[int] = archive_max_size
        #: the archive prune limit
        self._archive_prune_limit: Final[int] = archive_prune_limit
        #: the current archive size
        self._archive_size: int = 0
        #: the internal archive (pre-allocated to the prune limit)
        self._archive: Final[list[MORecord]] = []

    def _after_init(self) -> None:
        self._archive.extend(
            MORecord(self.create(), self.f_create())
            for _ in range(self._archive_prune_limit))
        super()._after_init()

    def check_in(self, x: Any, fs: np.ndarray,
                 prune_if_necessary: bool = False) -> bool:
        """
        Check a solution into the archive.

        :param x: the point in the search space
        :param fs: the vector of objective values
        :param prune_if_necessary: should we prune the archive if it becomes
            too large? `False` means that the archive may grow unbounded
        :returns: `True` if the solution was non-dominated, `False` if it was
            dominated by at least one solution in the archive
        """
        archive: Final[list[MORecord]] = self._archive
        added_to_archive: bool = False
        archive_size: int = self._archive_size
        # we update the archive
        domination: Final[Callable[[np.ndarray, np.ndarray], int]] \
            = self.f_dominates
        for i in range(archive_size - 1, -1, -1):
            ae: MORecord = archive[i]
            d: int = domination(fs, ae.fs)
            if d < 0:  # the new solution dominates an archived one
                if added_to_archive:  # if already added, shrink archive
                    archive_size = archive_size - 1
                    archive[archive_size], archive[i] = \
                        ae, archive[archive_size]
                else:  # if not added, overwrite dominated solution
                    self.copy(ae.x, x)
                    copyto(ae.fs, fs)
                    added_to_archive = True
            elif d > 0:
                return False

        if added_to_archive:  # already added, can quit
            self._archive_size = archive_size
        else:  # still need to add
            if archive_size >= len(archive):
                ae = MORecord(self.create(), self.f_create())
                archive.append(ae)
            else:
                ae = archive[archive_size]
            self.copy(ae.x, x)
            copyto(ae.fs, fs)
            archive_size += 1
            if prune_if_necessary \
                    and (archive_size > self._archive_prune_limit):
                self._prune(archive, self._archive_max_size, archive_size)
                self._archive_size = self._archive_max_size
            else:
                self._archive_size = archive_size
        return True

    def f_evaluate(self, x, fs: np.ndarray) -> float | int:
        if self._terminated:
            if self._knows_that_terminated:
                raise ValueError("The process has been terminated and "
                                 "the algorithm knows it.")
            return self._current_best_f

        result: Final[int | float] = self._f_evaluate(x, fs)
        self._current_fes = current_fes = self._current_fes + 1
        do_term: bool = current_fes >= self._end_fes

        improved: bool = False
        if result < self._current_best_f:
            improved = True
            self._current_best_f = result
            copyto(self._current_best_fs, fs)
            self._copy_y(self._current_best_y, x)
            do_term = do_term or (result <= self._end_f)

        if self.check_in(x, fs, True) or improved:
            self._last_improvement_fe = current_fes
            self._current_time_nanos = ctn = _TIME_IN_NS()
            self._last_improvement_time_nanos = ctn

        if do_term:
            self.terminate()

        return result

    def evaluate(self, x) -> float | int:
        return self.f_evaluate(x, self._fs_temp)

    def register(self, x, f: int | float) -> None:
        raise ValueError(
            "register is not supported in multi-objective optimization")

    def get_archive(self) -> list[MORecord]:
        return self._archive[0:self._archive_size]

    def get_copy_of_best_fs(self, fs: np.ndarray) -> None:
        if self._current_fes > 0:
            return copyto(fs, self._current_best_fs)
        raise ValueError("No current best available.")

    def _log_own_parameters(self, logger: KeyValueLogSection) -> None:
        super()._log_own_parameters(logger)
        logger.key_value(KEY_ARCHIVE_MAX_SIZE, self._archive_max_size)
        logger.key_value(KEY_ARCHIVE_PRUNE_LIMIT, self._archive_prune_limit)

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        _ProcessBase.log_parameters_to(self, logger)
        with logger.scope(SCOPE_PRUNER) as sc:
            self._pruner.log_parameters_to(sc)

    def _log_best(self, kv: KeyValueLogSection) -> None:
        super()._log_best(kv)
        kv.key_value(KEY_BEST_FS, array_to_str(self._current_best_fs))
        kv.key_value(KEY_ARCHIVE_SIZE, self._archive_size)

    def _log_and_check_archive_entry(self, index: int, rec: MORecord,
                                     logger: Logger) -> int | float:
        """
        Write an archive entry.

        :param index: the index of the entry
        :param rec: the record to verify
        :param logger: the logger
        :returns: the objective value
        """
        self.f_validate(rec.fs)
        self.validate(rec.x)
        tfs: Final[np.ndarray] = self._fs_temp
        f: Final[int | float] = self._f_evaluate(rec.x, tfs)
        if not np.array_equal(tfs, rec.fs):
            raise ValueError(
                f"expected {rec.fs} but got {tfs} when re-evaluating {rec}")
        if not isinstance(f, (int, float)):
            raise type_error(f, "scalarized objective value", (int, float))
        if not isfinite(f):
            raise ValueError(f"scalarized objective value {f} is not finite")

        with logger.text(f"{PREFIX_SECTION_ARCHIVE}{index}"
                         f"{SUFFIX_SECTION_ARCHIVE_Y}") as lg:
            lg.write(self.to_str(rec.x))
        return f

    def _write_log(self, logger: Logger) -> None:
        super()._write_log(logger)

        if self._archive_size > 0:
            # write and verify the archive
            archive: Final[list[MORecord]] = \
                self._archive[0:self._archive_size]
            archive.sort()
            qualities: Final[list[list[int | float]]] = []
            for i, rec in enumerate(archive):
                q: list[int | float] = [
                    np_to_py_number(n) for n in rec.fs]
                q.insert(0, self._log_and_check_archive_entry(i, rec, logger))
                qualities.append(q)

            # now write the qualities
            headline: list[str] = [
                f"{KEY_ARCHIVE_F}{i}" for i in range(self.f_dimension())]
            headline.insert(0, KEY_ARCHIVE_F)
            with logger.csv(SECTION_ARCHIVE_QUALITY, headline) as csv:
                for qq in qualities:
                    csv.row(qq)

    def _write_mo_log(self,
                      log: list[list[int | float | np.ndarray]],
                      start_time: int,
                      keep_all: bool,
                      logger: Logger) -> None:
        """
        Write the multi-objective log to the logger.

        :param log: the log
        :param start_time: the start time
        :param keep_all: do we need to keep all entries?
        :param logger: the destination logger
        """
        loglen = len(log)
        if loglen <= 0:
            return

        domination: Final[Callable[[np.ndarray, np.ndarray], int]] \
            = self.f_dominates

        if not keep_all:
            # first we clean the log from potentially dominated entries
            for i in range(loglen - 1, 0, -1):
                reci = log[i]
                fi = cast(int | float, reci[2])
                fsi = cast(np.ndarray, reci[3])
                for j in range(i - 1, -1, -1):
                    recj = log[j]
                    fj = cast(int | float, recj[2])
                    fsj = cast(np.ndarray, recj[3])
                    if (fj <= fi) and (domination(fsi, fsj) > 0):
                        del log[i]
                        break

        header: list[str] = [PROGRESS_FES, PROGRESS_TIME_MILLIS,
                             PROGRESS_CURRENT_F]
        for i in range(len(cast(np.ndarray, log[0])[3])):
            header.append(f"{PROGRESS_CURRENT_F}{i}")

        with logger.csv(SECTION_PROGRESS, header) as csv:
            for row in log:
                srow = [row[0], _ns_to_ms(cast(int, row[1])
                                          - start_time), row[2]]
                srow.extend([np_to_py_number(n)
                             for n in cast(np.ndarray, row[3])])
                csv.row(srow)

    def __str__(self) -> str:
        return "MOProcessWithoutSearchSpace"
