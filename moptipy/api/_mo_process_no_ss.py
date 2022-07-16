"""Providing a multi-objective process without logging with a single space."""

from math import isfinite, inf
from typing import Optional, Union, Final, Callable, Any, List

import numpy as np

from moptipy.api._process_base import _ProcessBase, _TIME_IN_NS
from moptipy.api.algorithm import Algorithm
from moptipy.api.logging import SCOPE_PRUNER, KEY_ARCHIVE_MAX_SIZE, \
    KEY_ARCHIVE_PRUNE_LIMIT, KEY_BEST_FS, SECTION_ARCHIVE_QUALITY, \
    KEY_ARCHIVE_F, PREFIX_SECTION_ARCHIVE, SUFFIX_SECTION_ARCHIVE_Y
from moptipy.api.mo_archive import MOArchivePruner, \
    check_mo_archive_pruner, MORecordY
from moptipy.api.mo_problem import MOProblem
from moptipy.api.mo_process import MOProcess
from moptipy.api.mo_utils import domination
from moptipy.api.space import Space
from moptipy.utils.logger import KeyValueLogSection, Logger
from moptipy.utils.nputils import np_number_to_py_number
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
                 log_file: Optional[Path] = None,
                 rand_seed: Optional[int] = None,
                 max_fes: Optional[int] = None,
                 max_time_millis: Optional[int] = None,
                 goal_f: Union[int, float, None] = None) -> None:
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
        self.f_copy = objective.f_copy  # type: ignore
        self.f_create = objective.f_create  # type: ignore
        self.f_to_str = objective.f_to_str  # type: ignore
        self.f_validate = objective.f_validate  # type: ignore

        #: the internal evaluation function
        self._f_evaluate: Final[Callable[
            [Any, np.ndarray], Union[int, float]]] = objective.f_evaluate

        #: the temporary variable for objective function evaluations
        self._fs_temp: Final[np.ndarray] = self.f_create()
        #: the holder for the objective vector of the current best solution
        self._current_best_fs: Final[np.ndarray] = self.f_create()
        #: the internal archive pruner
        self._pruner: Final[MOArchivePruner] = check_mo_archive_pruner(pruner)
        #: the fast call to the pruning routine
        self._prune: Final[Callable[[List[MORecordY], int], None]] \
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
        self._archive: Final[List[MORecordY]] = [
            MORecordY(self.f_create(), self.create())
            for _ in range(self._archive_prune_limit)
        ]

    def f_evaluate(self, x, fs: np.ndarray) -> Union[float, int]:
        if self._terminated:
            if self._knows_that_terminated:
                raise ValueError('The process has been terminated and '
                                 'the algorithm knows it.')
            return self._current_best_f

        result: Final[Union[int, float]] = self._f_evaluate(x, fs)
        self._current_fes = current_fes = self._current_fes + 1
        do_term: bool = current_fes >= self._end_fes

        improved: bool = False
        if result < self._current_best_f:
            improved = True
            self._current_best_f = result
            self.f_copy(self._current_best_fs, fs)
            self._last_improvement_fe = current_fes
            self._copy_y(self._current_best_y, x)

        archive: Final[List[MORecordY]] = self._archive
        dominated: bool = False
        added_to_archive: bool = False
        archive_size: int = self._archive_size
        # we update the archive
        for i in range(archive_size, -1, -1):
            ae: MORecordY = archive[i]
            d: int = domination(fs, ae.fs)
            if d < 0:  # the new solution dominates an archived one
                improved = True  # that's an improvement
                if added_to_archive:  # if already added, shrink archive
                    archive_size = archive_size - 1
                    archive[archive_size], archive[i] = \
                        ae, archive[archive_size]
                else:  # if not added, overwrite dominated solution
                    self.f_copy(ae.fs, fs)
                    self._copy_y(ae.x, x)
                    ae.f = result
                    added_to_archive = True
            elif d > 0:
                dominated = True
                break
        if added_to_archive:
            self._archive_size = archive_size
        elif not dominated:
            improved = True  # that is also an improvement
            ae = archive[archive_size]
            self.f_copy(ae.fs, fs)
            self._copy_y(ae.x, x)
            ae.f = result
            archive_size += 1
            if archive_size > self._archive_prune_limit:
                self._prune(archive, self._archive_max_size)
            self._archive_size = self._archive_max_size

        if improved:
            self._current_time_nanos = ctn = _TIME_IN_NS()
            self._last_improvement_time_nanos = ctn
            do_term = do_term or (result <= self._end_f)

        if do_term:
            self.terminate()

        return result

    def evaluate(self, x) -> Union[float, int]:
        return self.f_evaluate(x, self._fs_temp)

    def register(self, x, f: Union[int, float]) -> None:
        raise ValueError(
            "register is not supported in multi-objective optimization")

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        super(_ProcessBase, self).log_parameters_to(logger)
        logger.key_value(KEY_ARCHIVE_MAX_SIZE, self._archive_max_size)
        logger.key_value(KEY_ARCHIVE_PRUNE_LIMIT, self._archive_prune_limit)
        with logger.scope(SCOPE_PRUNER) as sc:
            self._pruner.log_parameters_to(sc)

    def _log_best(self, kv: KeyValueLogSection) -> None:
        super()._log_best(kv)
        kv.key_value(KEY_BEST_FS, self.f_to_str(self._current_best_fs))

    def _log_and_verify_archive_entry(self, index: int, rec: MORecordY,
                                      logger: Logger) -> None:
        """
        Write an archive entry.

        :param index: the index of the entry
        :param rec: the record to verify
        :param logger: the logger
        """
        tfs: Final[np.ndarray] = self._fs_temp
        f: Final[Union[int, float]] = self._f_evaluate(rec.x, tfs)
        if not np.array_equal(tfs, rec.fs):
            raise ValueError(
                f"expected {rec.fs} but got {tfs} when re-evaluating {rec}")
        if not isinstance(f, (int, float)):
            raise type_error(f, "scalarized objective value", (int, float))
        if not isfinite(f):
            raise ValueError(f"scalaized objective value {f} is not finite")
        if not isinstance(rec.f, (int, float)):
            raise type_error(rec.f, "rec.f", (int, float))
        if isfinite(rec.f):
            if rec.f != f:
                raise ValueError(f"rec.f={rec.f}, but computed f={f}!")
        elif rec.f >= inf:
            rec.f = f
        else:
            raise ValueError(f"rec.f={rec.f} is invalid!")
        self.f_validate(rec.fs)
        self.validate(rec.x)

        with logger.text(f"{PREFIX_SECTION_ARCHIVE}{index}"
                         f"{SUFFIX_SECTION_ARCHIVE_Y}") as lg:
            lg.write(self.to_str(rec.x))

    def _write_log(self, logger: Logger) -> None:
        super()._write_log(logger)

        if self._archive_size > 0:
            # write and verify the archive
            archive: Final[List[MORecordY]] = \
                self._archive[0:self._archive_size]
            del self._archive
            archive.sort()
            qualities: Final[List[List[Union[int, float]]]] = []
            for i, rec in enumerate(archive):
                self._log_and_verify_archive_entry(i, rec, logger)
                q: List[Union[int, float]] = [
                    np_number_to_py_number(n) for n in rec.fs]
                q.insert(0, rec.f)
                qualities.append(q)

            # now write the qualities
            headline: List[str] = [
                f"{KEY_ARCHIVE_F}{i}" for i in range(self.f_dimension())]
            headline.insert(0, KEY_ARCHIVE_F)
            with logger.csv(SECTION_ARCHIVE_QUALITY, headline) as csv:
                for qq in qualities:
                    csv.row(qq)

    def __str__(self) -> str:
        return "MOProcessWithoutSearchSpace"
