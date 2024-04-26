"""
SampleStatistics aggregated over multiple instances of `EndResult`.

The :mod:`~moptipy.evaluation.end_results` records hold the final result of
a run of an optimization algorithm on a problem instance. Often, we do not
want to compare these single results directly, but instead analyze summary
statistics, such as the mean best objective value found. For this purpose,
:class:`EndStatistics` exists. It summarizes the singular results from the
runs into a record with the most important statistics.
"""
import argparse
import os.path
from dataclasses import dataclass
from math import ceil, inf
from typing import Any, Callable, Final, Iterable, cast

from pycommons.io.console import logger
from pycommons.io.csv import (
    SCOPE_SEPARATOR,
    csv_column,
    csv_column_or_none,
    csv_read,
    csv_scope,
    csv_select_scope,
    csv_select_scope_or_none,
    csv_str_or_none,
    csv_val_or_none,
    csv_write,
)
from pycommons.io.path import Path, file_path, line_writer
from pycommons.math.sample_statistics import (
    KEY_MEAN_ARITH,
    KEY_STDDEV,
    SampleStatistics,
    from_samples,
    from_single_value,
)
from pycommons.math.sample_statistics import CsvReader as StatReader
from pycommons.math.sample_statistics import CsvWriter as StatWriter
from pycommons.math.sample_statistics import getter as stat_getter
from pycommons.strings.string_conv import (
    num_or_none_to_str,
    str_to_num,
)
from pycommons.types import (
    check_int_range,
    reiterable,
    type_error,
    type_name_of,
)

from moptipy.api.logging import (
    KEY_ALGORITHM,
    KEY_BEST_F,
    KEY_GOAL_F,
    KEY_INSTANCE,
    KEY_LAST_IMPROVEMENT_FE,
    KEY_LAST_IMPROVEMENT_TIME_MILLIS,
    KEY_MAX_FES,
    KEY_MAX_TIME_MILLIS,
    KEY_TOTAL_FES,
    KEY_TOTAL_TIME_MILLIS,
)
from moptipy.evaluation._utils import (
    _check_max_time_millis,
)
from moptipy.evaluation.base import (
    DESC_ALGORITHM,
    DESC_ENCODING,
    DESC_INSTANCE,
    DESC_OBJECTIVE_FUNCTION,
    F_NAME_RAW,
    F_NAME_SCALED,
    KEY_ENCODING,
    KEY_N,
    KEY_OBJECTIVE_FUNCTION,
    MultiRunData,
    _csv_motipy_footer,
)
from moptipy.evaluation.end_results import (
    DESC_BEST_F,
    DESC_GOAL_F,
    DESC_LAST_IMPROVEMENT_FE,
    DESC_LAST_IMPROVEMENT_TIME_MILLIS,
    DESC_MAX_FES,
    DESC_MAX_TIME_MILLIS,
    DESC_TOTAL_FES,
    DESC_TOTAL_TIME_MILLIS,
    EndResult,
)
from moptipy.evaluation.end_results import from_csv as end_results_from_csv
from moptipy.evaluation.end_results import from_logs as end_results_from_logs
from moptipy.utils.help import moptipy_argparser
from moptipy.utils.math import try_int, try_int_div

#: The key for the best F.
KEY_BEST_F_SCALED: Final[str] = KEY_BEST_F + "scaled"
#: The key for the number of successful runs.
KEY_N_SUCCESS: Final[str] = "successN"
#: The key for the success FEs.
KEY_SUCCESS_FES: Final[str] = "successFEs"
#: The key for the success time millis.
KEY_SUCCESS_TIME_MILLIS: Final[str] = "successTimeMillis"
#: The key for the ERT in FEs.
KEY_ERT_FES: Final[str] = "ertFEs"
#: The key for the ERT in milliseconds.
KEY_ERT_TIME_MILLIS: Final[str] = "ertTimeMillis"


@dataclass(frozen=True, init=False, order=False, eq=False)
class EndStatistics(MultiRunData):
    """
    Statistics over end results of one or multiple algorithm*instance setups.

    If one algorithm*instance is used, then `algorithm` and `instance` are
    defined. Otherwise, only the parameter which is the same over all recorded
    runs is defined.
    """

    #: The statistics about the best encountered result.
    best_f: SampleStatistics
    #: The statistics about the last improvement FE.
    last_improvement_fe: SampleStatistics
    #: The statistics about the last improvement time.
    last_improvement_time_millis: SampleStatistics
    #: The statistics about the total number of FEs.
    total_fes: SampleStatistics
    #: The statistics about the total time.
    total_time_millis: SampleStatistics
    #: The goal objective value.
    goal_f: SampleStatistics | int | float | None
    #: best_f / goal_f if goal_f is consistently defined and always positive.
    best_f_scaled: SampleStatistics | None
    #: The number of successful runs, if goal_f != None, else None.
    n_success: int | None
    #: The FEs to success, if n_success > 0, None otherwise.
    success_fes: SampleStatistics | None
    #: The time to success, if n_success > 0, None otherwise.
    success_time_millis: SampleStatistics | None
    #: The ERT if FEs, while is inf if n_success=0, None if goal_f is None,
    #: and finite otherwise.
    ert_fes: int | float | None
    #: The ERT if milliseconds, while is inf if n_success=0, None if goal_f
    #: is None, and finite otherwise.
    ert_time_millis: int | float | None
    #: The budget in FEs, if every run had one; None otherwise.
    max_fes: SampleStatistics | int | None
    #: The budget in milliseconds, if every run had one; None otherwise.
    max_time_millis: SampleStatistics | int | None

    def __init__(self,
                 algorithm: str | None,
                 instance: str | None,
                 objective: str | None,
                 encoding: str | None,
                 n: int,
                 best_f: SampleStatistics,
                 last_improvement_fe: SampleStatistics,
                 last_improvement_time_millis: SampleStatistics,
                 total_fes: SampleStatistics,
                 total_time_millis: SampleStatistics,
                 goal_f: float | int | SampleStatistics | None,
                 best_f_scaled: SampleStatistics | None,
                 n_success: int | None,
                 success_fes: SampleStatistics | None,
                 success_time_millis: SampleStatistics | None,
                 ert_fes: int | float | None,
                 ert_time_millis: int | float | None,
                 max_fes: SampleStatistics | int | None,
                 max_time_millis: SampleStatistics | int | None):
        """
        Create the end statistics of an experiment-setup combination.

        :param algorithm: the algorithm name, if all runs are with the same
            algorithm
        :param instance: the instance name, if all runs are on the same
            instance
        :param objective: the objective name, if all runs are on the same
            objective function, `None` otherwise
        :param encoding: the encoding name, if all runs are on the same
            encoding and an encoding was actually used, `None` otherwise
        :param n: the total number of runs
        :param best_f: statistics about the best achieved result
        :param last_improvement_fe: statistics about the last improvement FE
        :param last_improvement_time_millis: statistics about the last
            improvement time
        :param total_fes: statistics about the total FEs
        :param total_time_millis: statistics about the total runtime in
            milliseconds
        :param goal_f: if the goal objective value is not defined sometimes,
            this will be `None`. If it is always defined and always the same,
            then this will be that value. If different goal values exist, then
            this is the `SampleStatistics` record about them
        :param best_f_scaled: if `goal_f` is not `None` and greater than zero,
            then here we provide statistics about `best_f` divided by the
            corresponding `goal_f`
        :param n_success: the number of successful runs is only defined if
            `goal_f` is not `None` and counts the number of runs that reach or
            surpass their corresponding `goal_f`
        :param success_fes: if `goal_f` is not `None`,
            then this holds statistics about the last improvement FE of only
            the successful runs
        :param success_time_millis: if `goal_f` is not `None`, then this holds
            statistics about the last improvement times of only the successful
            runs
        :param ert_fes: if `goal_f` is always defined, then this is the
            empirically estimated running time to solve the problem in FEs if
            `n_success>0` and `inf` otherwise
        :param ert_time_millis: if `goal_f` is always defined, then this is
            the empirically estimated running time to solve the problem in
            milliseconds if `n_success>0` and `inf` otherwise
        :param max_fes: the budget in FEs, if any
        :param max_time_millis: the budget in terms of milliseconds
        """
        super().__init__(algorithm, instance, objective, encoding, n)

        if not isinstance(best_f, SampleStatistics):
            raise type_error(best_f, "best_f", SampleStatistics)
        object.__setattr__(self, "best_f", best_f)
        if best_f.n != n:
            raise ValueError(f"best_f.n={best_f.n} != n={n}")

        if not isinstance(last_improvement_fe, SampleStatistics):
            raise type_error(last_improvement_fe, "last_improvement_fe",
                             SampleStatistics)
        if last_improvement_fe.n != n:
            raise ValueError(
                f"last_improvement_fe.n={last_improvement_fe.n} != n={n}")
        check_int_range(
            last_improvement_fe.minimum, "last_improvement_fe.minimum",
            1, 1_000_000_000_000_000)
        check_int_range(
            last_improvement_fe.maximum, "last_improvement_fe.maximum",
            last_improvement_fe.minimum, 1_000_000_000_000_000)
        object.__setattr__(self, "last_improvement_fe", last_improvement_fe)

        if not isinstance(last_improvement_time_millis, SampleStatistics):
            raise type_error(last_improvement_time_millis,
                             "last_improvement_time_millis", SampleStatistics)
        if last_improvement_time_millis.n != n:
            raise ValueError("last_improvement_time_millis.n="
                             f"{last_improvement_time_millis.n} != n={n}")
        check_int_range(
            last_improvement_time_millis.minimum,
            "last_improvement_time_millis.minimum",
            0, 100_000_000_000)
        check_int_range(
            last_improvement_time_millis.maximum,
            "last_improvement_time_millis.maximum",
            last_improvement_time_millis.minimum, 100_000_000_000)
        object.__setattr__(self, "last_improvement_time_millis",
                           last_improvement_time_millis)

        if not isinstance(total_fes, SampleStatistics):
            raise type_error(total_fes, "total_fes", SampleStatistics)
        if total_fes.n != n:
            raise ValueError(
                f"total_fes.n={total_fes.n} != n={n}")
        check_int_range(
            total_fes.minimum, "total_fes.minimum",
            last_improvement_fe.minimum, 1_000_000_000_000_000)
        check_int_range(
            total_fes.maximum, "total_fes.maximum",
            max(total_fes.minimum, last_improvement_fe.maximum),
            1_000_000_000_000_000)
        object.__setattr__(self, "total_fes", total_fes)

        if not isinstance(total_time_millis, SampleStatistics):
            raise type_error(total_time_millis, "total_time_millis",
                             SampleStatistics)
        if total_time_millis.n != n:
            raise ValueError(
                f"total_time_millis.n={total_time_millis.n} != n={n}")
        check_int_range(
            total_time_millis.minimum, "total_time_millis.minimum",
            last_improvement_time_millis.minimum, 100_000_000_000)
        check_int_range(
            total_time_millis.maximum, "total_time_millis.maximum",
            max(total_time_millis.minimum,
                last_improvement_time_millis.maximum),
            100_000_000_000)
        object.__setattr__(self, "total_time_millis", total_time_millis)

        if goal_f is None:
            if best_f_scaled is not None:
                raise ValueError(
                    "If goal_f is None, best_f_scaled must also be None, "
                    f"but is {type(best_f_scaled)}.")
            if n_success is not None:
                raise ValueError(
                    "If goal_f is None, n_success must also be None, "
                    f"but is {type(n_success)}.")
            if success_fes is not None:
                raise ValueError(
                    "If success_fes is None, best_f_scaled must also be None, "
                    f"but is {type(success_fes)}.")
            if success_time_millis is not None:
                raise ValueError(
                    "If success_time_millis is None, best_f_scaled "
                    "must also be None, "
                    f"but is {type(success_time_millis)}.")
            if ert_fes is not None:
                raise ValueError(
                    "If goal_f is None, ert_fes must also be None, "
                    f"but is {type(ert_fes)}.")
            if ert_time_millis is not None:
                raise ValueError(
                    "If goal_f is None, ert_time_millis must also be None, "
                    f"but is {type(ert_time_millis)}.")
        else:  # goal_f is not None
            if isinstance(goal_f, SampleStatistics):
                if goal_f.n != n:
                    raise ValueError(f"goal_f.n={goal_f.n} != n={n}")
                goal_f = goal_f.compact(False)
            if isinstance(goal_f, float):
                goal_f = None if goal_f <= (-inf) else try_int(goal_f)
            elif not isinstance(goal_f, int | SampleStatistics):
                raise type_error(goal_f, "goal_f", (
                    int, float, SampleStatistics))

            if best_f_scaled is not None:
                goal_f_min: Final[int | float] = \
                    goal_f.minimum if isinstance(goal_f, SampleStatistics) \
                    else goal_f
                if goal_f_min <= 0:
                    raise ValueError(
                        f"best_f_scaled must be None if minimum goal_f "
                        f"({goal_f_min}) of goal_f {goal_f} is not positive,"
                        f" but is {best_f_scaled}.")
                if not isinstance(best_f_scaled, SampleStatistics):
                    raise type_error(best_f_scaled, "best_f_scaled",
                                     SampleStatistics)
                if best_f_scaled.n != n:
                    raise ValueError(
                        f"best_f_scaled.n={best_f_scaled.n} != n={n}")
                if best_f_scaled.minimum < 0:
                    raise ValueError(
                        "best_f_scaled cannot be negative, but encountered "
                        f"{best_f_scaled.minimum}.")

            check_int_range(n_success, "n_success")
            if not isinstance(ert_fes, int | float):
                raise type_error(ert_fes, "ert_fes", (int, float))
            if not isinstance(ert_time_millis, int | float):
                raise type_error(ert_time_millis, "ert_time_millis",
                                 (int, float))

            if n_success > 0:
                if not isinstance(success_fes, SampleStatistics):
                    raise type_error(success_fes,
                                     "if n_success>0, then success_fes",
                                     SampleStatistics)
                if success_fes.n != n_success:
                    raise ValueError(f"success_fes.n={success_fes.n} != "
                                     f"n_success={n_success}")
                check_int_range(
                    success_fes.minimum, "success_fes.minimum",
                    last_improvement_fe.minimum, 1_000_000_000_000_000)
                check_int_range(
                    success_fes.maximum, "success_fes.maximum",
                    success_fes.minimum, last_improvement_fe.maximum)
                if not isinstance(success_time_millis, SampleStatistics):
                    raise type_error(
                        success_time_millis,
                        "if n_success>0, then success_time_millis",
                        SampleStatistics)
                if success_time_millis.n != n_success:
                    raise ValueError(
                        f"success_time_millis.n={success_time_millis.n} != "
                        f"n_success={n_success}")
                check_int_range(
                    success_time_millis.minimum,
                    "success_time_millis.minimum",
                    last_improvement_time_millis.minimum, 100_000_000_000)
                check_int_range(
                    success_time_millis.maximum,
                    "success_time_millis.maximum",
                    success_time_millis.minimum,
                    last_improvement_time_millis.maximum)
                ert_fes = try_int(ert_fes)
                if ert_fes < success_fes.minimum:
                    raise ValueError(
                        "ert_fes must be >= "
                        f"{success_fes.minimum}, but is {ert_fes}.")
                ert_fe_max = ceil(total_fes.mean_arith * n)
                if ert_fes > ert_fe_max:
                    raise ValueError(
                        "ert_fes must be <= "
                        f"{ert_fe_max}, but is {ert_fes}.")

                ert_time_millis = try_int(ert_time_millis)
                if ert_time_millis < success_time_millis.minimum:
                    raise ValueError(
                        "ert_time_millis must be >= "
                        f"{success_time_millis.minimum}, but "
                        f"is {ert_time_millis}.")
                ert_time_max = ceil(total_time_millis.mean_arith * n)
                if ert_time_millis > ert_time_max:
                    raise ValueError(
                        "ert_time_millis must be <= "
                        f"{ert_time_max}, but is {ert_time_millis}.")
            else:
                if success_fes is not None:
                    raise ValueError(
                        "If n_success<=0, then success_fes must be None, "
                        f"but it's a {type_name_of(success_fes)}.")
                if success_time_millis is not None:
                    raise ValueError(
                        "If n_success<=0, then success_time_millis must be "
                        f"None, but it is a "
                        f"{type_name_of(success_time_millis)}.")
                if ert_fes < inf:
                    raise ValueError(
                        "If n_success<=0, then ert_fes must "
                        f"be inf, but it's {ert_fes}.")
                if ert_time_millis < inf:
                    raise ValueError(
                        "If n_success<=0, then ert_time_millis must "
                        f"be inf, but it's {ert_time_millis}.")

        object.__setattr__(self, "goal_f", goal_f)
        object.__setattr__(self, "best_f_scaled", best_f_scaled)
        object.__setattr__(self, "n_success", n_success)
        object.__setattr__(self, "success_fes", success_fes)
        object.__setattr__(self, "success_time_millis", success_time_millis)
        object.__setattr__(self, "ert_fes", ert_fes)
        object.__setattr__(self, "ert_time_millis", ert_time_millis)

        if isinstance(max_fes, SampleStatistics):
            if max_fes.n != n:
                raise ValueError(f"max_fes.n={max_fes.n} != n={n}")
            max_fes_f: int | float | SampleStatistics = max_fes.compact(
                needs_n=False)
            if isinstance(max_fes_f, float):
                raise type_error(max_fes_f, "max_fes", (
                    int, SampleStatistics, None))
            max_fes = max_fes_f
        if isinstance(max_fes, int):
            if (max_fes < total_fes.maximum) or (max_fes < 0):
                raise ValueError(f"0<max_fes must be >= "
                                 f"{total_fes.maximum}, but is {max_fes}.")
        elif isinstance(max_fes, SampleStatistics):
            if (max_fes.minimum < total_fes.minimum) or (
                    max_fes.minimum <= 0):
                raise ValueError(
                    f"0<max_fes.minimum must be >= {total_fes.minimum},"
                    f" but is {max_fes.minimum}.")
            if max_fes.maximum < total_fes.maximum:
                raise ValueError(
                    f"max_fes.maximum must be >= {total_fes.maximum},"
                    f" but is {max_fes.maximum}.")
        elif max_fes is not None:
            raise type_error(max_fes, "max_fes", (int, SampleStatistics, None))
        object.__setattr__(self, "max_fes", max_fes)

        if isinstance(max_time_millis, SampleStatistics):
            if max_time_millis.n != n:
                raise ValueError(
                    f"max_time_millis.n={max_time_millis.n} != n={n}")
            max_time_millis_f: int | float | SampleStatistics = (
                max_time_millis.compact(False))
            if isinstance(max_time_millis_f, float):
                raise type_error(max_time_millis_f, "max_time_millis", (
                    int, SampleStatistics, None))
        if isinstance(max_time_millis, int):
            _check_max_time_millis(max_time_millis,
                                   total_fes.minimum,
                                   total_time_millis.maximum)
        elif isinstance(max_time_millis, SampleStatistics):
            _check_max_time_millis(max_time_millis.minimum,
                                   total_fes.minimum,
                                   total_time_millis.minimum)
            _check_max_time_millis(max_time_millis.maximum,
                                   total_fes.minimum,
                                   total_time_millis.maximum)
        elif max_time_millis is not None:
            raise type_error(max_time_millis, "max_time_millis",
                             (int, SampleStatistics, None))
        object.__setattr__(self, "max_time_millis", max_time_millis)

    def get_n(self) -> int:
        """
        Get the number of runs.

        :returns: the number of runs.
        """
        if not isinstance(self, EndStatistics):
            raise type_error(self, "self", EndStatistics)
        return self.n

    def get_best_f(self) -> SampleStatistics:
        """
        Get the statistics about the best objective value reached.

        :returns: the statistics about the best objective value reached
        """
        if not isinstance(self, EndStatistics):
            raise type_error(self, "self", EndStatistics)
        return self.best_f

    def get_last_improvement_fe(self) -> SampleStatistics:
        """
        Get the statistics about the last improvement FE.

        :returns: the statistics about the last improvement FE
        """
        if not isinstance(self, EndStatistics):
            raise type_error(self, "self", EndStatistics)
        return self.last_improvement_fe

    def get_last_improvement_time_millis(self) -> SampleStatistics:
        """
        Get the statistics about the last improvement time millis.

        :returns: the statistics about the last improvement time millis
        """
        if not isinstance(self, EndStatistics):
            raise type_error(self, "self", EndStatistics)
        return self.last_improvement_time_millis

    def get_total_fes(self) -> SampleStatistics:
        """
        Get the statistics about the total FEs.

        :returns: the statistics about the total FEs
        """
        if not isinstance(self, EndStatistics):
            raise type_error(self, "self", EndStatistics)
        return self.total_fes

    def get_total_time_millis(self) -> SampleStatistics:
        """
        Get the statistics about the total time millis.

        :returns: the statistics about the total time millis
        """
        if not isinstance(self, EndStatistics):
            raise type_error(self, "self", EndStatistics)
        return self.total_time_millis

    def get_goal_f(self) -> SampleStatistics | int | float | None:
        """
        Get the statistics about the goal objective value.

        :returns: the statistics about the goal objective value
        """
        if not isinstance(self, EndStatistics):
            raise type_error(self, "self", EndStatistics)
        return self.goal_f

    def get_best_f_scaled(self) -> SampleStatistics | None:
        """
        Get the statistics about the scaled best objective value.

        :returns: the statistics about the scaled best objective value
        """
        if not isinstance(self, EndStatistics):
            raise type_error(self, "self", EndStatistics)
        return self.best_f_scaled

    def get_n_success(self) -> int | None:
        """
        Get the number of successful runs.

        :returns: the number of successful runs.
        """
        if not isinstance(self, EndStatistics):
            raise type_error(self, "self", EndStatistics)
        return self.n_success

    def get_success_fes(self) -> SampleStatistics | None:
        """
        Get the statistics about the FEs until success of the successful runs.

        :returns: the statistics about the FEs until success of the successful
            runs
        """
        if not isinstance(self, EndStatistics):
            raise type_error(self, "self", EndStatistics)
        return self.success_fes

    def get_success_time_millis(self) -> SampleStatistics | None:
        """
        Get the statistics about the ms until success of the successful runs.

        :returns: the statistics about the ms until success of the successful
            runs
        """
        if not isinstance(self, EndStatistics):
            raise type_error(self, "self", EndStatistics)
        return self.success_time_millis

    def get_ert_fes(self) -> int | float | None:
        """
        Get the expected FEs until success.

        :returns: the statistics about the expected FEs until success.
        """
        if not isinstance(self, EndStatistics):
            raise type_error(self, "self", EndStatistics)
        return self.ert_fes

    def get_ert_time_millis(self) -> int | float | None:
        """
        Get the expected milliseconds until success.

        :returns: the statistics about the expected milliseconds until
            success.
        """
        if not isinstance(self, EndStatistics):
            raise type_error(self, "self", EndStatistics)
        return self.ert_time_millis

    def get_max_fes(self) -> SampleStatistics | int | None:
        """
        Get the statistics about the maximum permitted FEs.

        :returns: the statistics about the maximum permitted FEs
        """
        if not isinstance(self, EndStatistics):
            raise type_error(self, "self", EndStatistics)
        return self.max_fes

    def get_max_time_millis(self) -> SampleStatistics | int | None:
        """
        Get the statistics about the maximum permitted runtime in ms.

        :returns: the statistics about the maximum permitted runtime in ms
        """
        if not isinstance(self, EndStatistics):
            raise type_error(self, "self", EndStatistics)
        return self.max_time_millis


def create(source: Iterable[EndResult]) -> EndStatistics:
    """
    Create an `EndStatistics` Record from an Iterable of `EndResult`.

    :param source: the source
    :return: the statistics
    :rtype: EndStatistics
    """
    if not isinstance(source, Iterable):
        raise type_error(source, "source", Iterable)

    n: int = 0
    best_f: list[int | float] = []
    last_improvement_fe: list[int] = []
    last_improvement_time_millis: list[int] = []
    total_fes: list[int] = []
    total_time_millis: list[int] = []
    max_fes: list[int] | None = []
    max_fes_same: bool = True
    max_time_millis: list[int] | None = []
    max_time_same: bool = True
    goal_f: list[int | float] | None = []
    goal_f_same: bool = True
    best_f_scaled: list[float] | None = []
    n_success: int | None = 0
    success_fes: list[int] | None = []
    success_times: list[int] | None = []

    fes: int = 0
    time: int = 0
    algorithm: str | None = None
    instance: str | None = None
    objective: str | None = None
    encoding: str | None = None

    for er in source:
        if not isinstance(er, EndResult):
            raise type_error(er, "end result", EndResult)
        if n == 0:
            algorithm = er.algorithm
            instance = er.instance
            objective = er.objective
            encoding = er.encoding
        else:
            if algorithm != er.algorithm:
                algorithm = None
            if instance != er.instance:
                instance = None
            if objective != er.objective:
                objective = None
            if encoding != er.encoding:
                encoding = None
        n += 1
        best_f.append(er.best_f)
        last_improvement_fe.append(er.last_improvement_fe)
        last_improvement_time_millis.append(
            er.last_improvement_time_millis)
        total_fes.append(er.total_fes)
        total_time_millis.append(er.total_time_millis)
        if er.max_fes is None:
            max_fes = None
        elif max_fes is not None:
            if n > 1:
                max_fes_same = max_fes_same \
                    and (max_fes[-1] == er.max_fes)
            max_fes.append(er.max_fes)
        if er.max_time_millis is None:
            max_time_millis = None
        elif max_time_millis is not None:
            if n > 1:
                max_time_same = \
                    max_time_same \
                    and (max_time_millis[-1] == er.max_time_millis)
            max_time_millis.append(er.max_time_millis)

        if er.goal_f is None:
            goal_f = None
            best_f_scaled = None
            n_success = None
            success_fes = None
            success_times = None
        elif goal_f is not None:
            if n > 1:
                goal_f_same = goal_f_same and (goal_f[-1] == er.goal_f)
            goal_f.append(er.goal_f)

            if er.goal_f <= 0:
                best_f_scaled = None
            elif best_f_scaled is not None:
                best_f_scaled.append(er.best_f / er.goal_f)

            if er.best_f <= er.goal_f:
                n_success += 1
                success_fes.append(er.last_improvement_fe)
                success_times.append(er.last_improvement_time_millis)
                fes += er.last_improvement_fe
                time += er.last_improvement_time_millis
            else:
                fes += er.total_fes
                time += er.total_time_millis
    if n <= 0:
        raise ValueError("There must be at least one end result record.")

    return EndStatistics(
        algorithm,
        instance,
        objective,
        encoding,
        n,
        from_samples(best_f),
        from_samples(last_improvement_fe),
        from_samples(last_improvement_time_millis),
        from_samples(total_fes),
        from_samples(total_time_millis),
        None if (goal_f is None)
        else (goal_f[0] if goal_f_same else from_samples(goal_f)),
        None if (best_f_scaled is None)
        else from_samples(best_f_scaled),
        n_success,
        None if (n_success is None) or (n_success <= 0)
        else from_samples(success_fes),
        None if (n_success is None) or (n_success <= 0)
        else from_samples(success_times),
        None if (n_success is None)
        else (inf if (n_success <= 0) else try_int_div(fes, n_success)),
        None if (n_success is None) else
        (inf if (n_success <= 0) else try_int_div(time, n_success)),
        None if max_fes is None else
        (max_fes[0] if max_fes_same else from_samples(max_fes)),
        None if max_time_millis is None
        else (max_time_millis[0] if max_time_same
              else from_samples(max_time_millis)))


def from_end_results(source: Iterable[EndResult],
                     consumer: Callable[[EndStatistics], Any],
                     join_all_algorithms: bool = False,
                     join_all_instances: bool = False,
                     join_all_objectives: bool = False,
                     join_all_encodings: bool = False) -> None:
    """
    Aggregate statistics over a stream of end results.

    :param source: the stream of end results
    :param consumer: the destination to which the new records will be
        sent, can be the `append` method of a :class:`list`
    :param join_all_algorithms: should the statistics be aggregated
        over all algorithms
    :param join_all_instances: should the statistics be aggregated
        over all algorithms
    :param join_all_objectives: should the statistics be aggregated over
        all objectives?
    :param join_all_encodings: should statistics be aggregated over all
        encodings
    """
    if not isinstance(source, Iterable):
        raise type_error(source, "source", Iterable)
    if not callable(consumer):
        raise type_error(consumer, "consumer", call=True)
    if not isinstance(join_all_algorithms, bool):
        raise type_error(join_all_algorithms,
                         "join_all_algorithms", bool)
    if not isinstance(join_all_instances, bool):
        raise type_error(join_all_instances, "join_all_instances", bool)
    if not isinstance(join_all_objectives, bool):
        raise type_error(join_all_objectives, "join_all_objectives", bool)
    if not isinstance(join_all_encodings, bool):
        raise type_error(join_all_encodings, "join_all_encodings", bool)

    if (join_all_algorithms and join_all_instances
            and join_all_objectives and join_all_encodings):
        consumer(create(source))
        return

    sorter: dict[tuple[str, str, str, str], list[EndResult]] = {}
    for er in source:
        if not isinstance(er, EndResult):
            raise type_error(source, "end results from source",
                             EndResult)
        key = ("" if join_all_algorithms else er.algorithm,
               "" if join_all_instances else er.instance,
               "" if join_all_objectives else er.objective,
               "" if join_all_encodings else (
                   "" if er.encoding is None else er.encoding))
        if key in sorter:
            lst = sorter[key]
        else:
            lst = []
            sorter[key] = lst
        lst.append(er)

    if len(sorter) <= 0:
        raise ValueError("source must not be empty")

    if len(sorter) > 1:
        for key in sorted(sorter.keys()):
            consumer(create(sorter[key]))
    else:
        consumer(create(next(iter(sorter.values()))))


def to_csv(data: EndStatistics | Iterable[EndStatistics],
           file: str) -> Path:
    """
    Store a set of :class:`EndStatistics` in a CSV file.

    :param data: the data to store
    :param file: the file to generate
    :return: the path to the generated CSV file
    """
    path: Final[Path] = Path(file)
    logger(f"Writing end result statistics to CSV file {path!r}.")
    path.ensure_parent_dir_exists()
    with path.open_for_write() as wt:
        csv_write(
            data=[data] if isinstance(data, EndStatistics) else sorted(data),
            consumer=line_writer(wt),
            setup=CsvWriter().setup,
            get_column_titles=CsvWriter.get_column_titles,
            get_row=CsvWriter.get_row,
            get_header_comments=CsvWriter.get_header_comments,
            get_footer_comments=CsvWriter.get_footer_comments)
    logger(f"Done writing end result statistics to CSV file {path!r}.")
    return path


def from_csv(file: str,
             consumer: Callable[[EndStatistics], Any]) -> None:
    """
    Parse a CSV file and collect all encountered :class:`EndStatistics`.

    :param file: the file to parse
    :param consumer: the consumer to receive all the parsed instances of
        :class:`~moptipy.evaluation.end_statistics.EndStatistics`, can be
        the `append` method of a :class:`list`
    """
    path: Final[Path] = file_path(file)
    logger(f"Begin reading end result statistics from CSV file {path!r}.")
    with path.open_for_read() as rd:
        csv_read(rows=rd,
                 setup=CsvReader,
                 parse_row=CsvReader.parse_row,
                 consumer=consumer)
    logger("Finished reading end result statistics from CSV "
           f"file {path!r}.")


#: the internal getters that can work directly
__PROPERTIES: Final[Callable[[str], Callable[[
    EndStatistics], SampleStatistics | int | float | None] | None]] = {
    KEY_N: EndStatistics.get_n,
    KEY_N_SUCCESS: EndStatistics.get_n_success,
    KEY_ERT_FES: EndStatistics.get_ert_fes,
    KEY_ERT_TIME_MILLIS: EndStatistics.get_ert_time_millis,
    KEY_GOAL_F: EndStatistics.get_goal_f,
    KEY_MAX_TIME_MILLIS: EndStatistics.get_max_time_millis,
    KEY_MAX_FES: EndStatistics.get_max_fes,
    KEY_BEST_F: EndStatistics.get_best_f,
    F_NAME_RAW: EndStatistics.get_best_f,
    KEY_LAST_IMPROVEMENT_FE: EndStatistics.get_last_improvement_fe,
    "last improvement FE": EndStatistics.get_last_improvement_fe,
    KEY_LAST_IMPROVEMENT_TIME_MILLIS:
        EndStatistics.get_last_improvement_time_millis,
    "last improvement ms": EndStatistics.get_last_improvement_time_millis,
    KEY_BEST_F_SCALED: EndStatistics.get_best_f_scaled,
    KEY_SUCCESS_FES: EndStatistics.get_success_fes,
    KEY_SUCCESS_TIME_MILLIS: EndStatistics.get_success_time_millis,
    F_NAME_SCALED: EndStatistics.get_best_f_scaled,
    KEY_TOTAL_FES: EndStatistics.get_total_fes,
    "fes": EndStatistics.get_total_fes,
    KEY_TOTAL_TIME_MILLIS: EndStatistics.get_total_time_millis,
    "ms": EndStatistics.get_total_time_millis,
    "f": EndStatistics.get_best_f,
    "budgetFEs": EndStatistics.get_max_fes,
    "budgetMS": EndStatistics.get_max_time_millis,
}.get

#: the success keys
__SUCCESS_KEYS: Final[Callable[[str], bool]] = {
    KEY_SUCCESS_FES, KEY_SUCCESS_TIME_MILLIS,
}.__contains__

#: the internal static getters
__STATIC: Final[dict[str, Callable[[EndStatistics], int | float | None]]] = {
    KEY_N: EndStatistics.get_n,
    KEY_N_SUCCESS: EndStatistics.get_n_success,
    KEY_ERT_FES: EndStatistics.get_ert_fes,
    KEY_ERT_TIME_MILLIS: EndStatistics.get_ert_time_millis,
}


def getter(dimension: str) -> Callable[[EndStatistics], int | float | None]:
    """
    Create a function that obtains the given dimension from EndStatistics.

    :param dimension: the dimension
    :returns: a callable that returns the value corresponding to the
        dimension
    """
    dimension = str.strip(dimension)
    direct: Callable[[EndStatistics], int | float | None] = \
        __STATIC.get(dimension)
    if direct is not None:
        return direct

    names: Final[list[str]] = str.split(str.strip(dimension), SCOPE_SEPARATOR)
    n_names: Final[int] = list.__len__(names)
    if not (0 < n_names < 3):
        raise ValueError(
            f"Invalid name combination {dimension!r} -> {names!r}.")
    getter_1: Final[Callable[[
        EndStatistics], int | float | SampleStatistics | None] | None] = \
        __PROPERTIES(names[0])
    if getter_1 is None:
        raise ValueError(f"Invalid dimension {names[0]!r} in {dimension!r}.")
    getter_2: Final[Callable[[
        SampleStatistics], int | float | None]] = \
        stat_getter(names[1] if n_names > 1 else KEY_MEAN_ARITH)

    if getter_2 is stat_getter(KEY_STDDEV):  # it is sd
        n_prop: Final[Callable[[EndStatistics], int | None]] = \
            EndStatistics.get_n_success if __SUCCESS_KEYS(
                names[0]) else EndStatistics.get_n

        def __combo_sd(
                data: EndStatistics, __g1=getter_1, __g2=getter_2,
                __n=n_prop) -> int | float | None:
            val: int | float | SampleStatistics | None = __g1(data)
            if val is None:
                return None
            if isinstance(val, int | float):
                n = __n(data)
                return None if (n is None) or (n <= 0) else 0
            return __g2(val)
        direct = cast(Callable[[EndStatistics], int | float | None],
                      __combo_sd)
    else:  # any other form of mean or statistic

        def __combo_no_sd(data: EndStatistics,
                          __g1=getter_1, __g2=getter_2) -> int | float | None:
            val: int | float | SampleStatistics | None = __g1(data)
            if (val is None) or (isinstance(val, int | float)):
                return val
            return __g2(val)
        direct = cast(Callable[[EndStatistics], int | float | None],
                      __combo_no_sd)

    __STATIC[dimension] = direct
    return direct


def _to_csv_writer(
        get_func: Callable[
            [EndStatistics], SampleStatistics | int | float | None],
        n_func: Callable[[EndStatistics], int],
        data: Iterable[EndStatistics],
        scope: str | None = None,
        what_short: str | None = None,
        what_long: str | None = None) -> StatWriter | None:
    """
    Get a CSV Writer for the given data subset.

    :param get_func: the getter for the value
    :param n_func: the n-getter
    :param data: the data iterator
    :param scope: the scope to use
    :param what_short: the short description
    :param what_long: the long description
    :returns: the writer, if there was any associated data
    """
    refined: list[tuple[SampleStatistics | int | float | None, int]] = [
        v for v in ((get_func(es), n_func(es)) for es in data)
        if v[0] is not None]
    if list.__len__(refined) <= 0:
        return None
    return StatWriter(scope=scope, n_not_needed=True, what_short=what_short,
                      what_long=what_long).setup((
                          from_single_value(v, n) for v, n in refined))


class CsvWriter:
    """A class for CSV writing of :class:`EndStatistics`."""

    def __init__(self, scope: str | None = None) -> None:
        """
        Initialize the csv writer.

        :param scope: the prefix to be pre-pended to all columns
        """
        #: an optional scope
        self.scope: Final[str | None] = (
            str.strip(scope)) if scope is not None else None

        #: has this writer been set up?
        self.__setup: bool = False
        #: do we put the algorithm column?
        self.__has_algorithm: bool = False
        #: do we put the instance column?
        self.__has_instance: bool = False
        #: do we put the objective column?
        self.__has_objective: bool = False
        #: do we put the encoding column?
        self.__has_encoding: bool = False
        #: do we put the goal_f column?
        self.__goal_f: StatWriter | None = None
        #: the best objective value reached
        self.__best_f: Final[StatWriter] = StatWriter(
            csv_scope(scope, KEY_BEST_F), True, KEY_BEST_F,
            "the best objective value reached per run")
        #: the FE when the last improvement happened
        self.__life: Final[StatWriter] = StatWriter(
            csv_scope(scope, KEY_LAST_IMPROVEMENT_FE), True,
            KEY_LAST_IMPROVEMENT_FE,
            "the FE when the last improvement happened in a run",
        )
        #: the milliseconds when the last improvement happened
        self.__lims: Final[StatWriter] = StatWriter(
            csv_scope(scope, KEY_LAST_IMPROVEMENT_TIME_MILLIS), True,
            KEY_LAST_IMPROVEMENT_TIME_MILLIS,
            "the millisecond when the last improvement happened in a run",
        )
        #: the total FEs
        self.__total_fes: Final[StatWriter] = StatWriter(
            csv_scope(scope, KEY_TOTAL_FES), True,
            KEY_TOTAL_FES,
            "the total FEs consumed by the runs",
        )
        #: the total milliseconds
        self.__total_ms: Final[StatWriter] = StatWriter(
            csv_scope(scope, KEY_TOTAL_TIME_MILLIS), True,
            KEY_TOTAL_TIME_MILLIS,
            "the total millisecond consumed by a run",
        )
        #: do we put the best-f-scaled column?
        self.__best_f_scaled: StatWriter | None = None
        #: do we put the n_success column?
        self.__has_n_success: bool = False
        #: do we put the success fes column?
        self.__success_fes: StatWriter | None = None
        #: do we put the success time millis column?
        self.__success_time_millis: StatWriter | None = None
        #: do we put the ert-fes column?
        self.__has_ert_fes: bool = False
        #: do we put the ert time millis column?
        self.__has_ert_time_millis: bool = False
        #: do we put the max-fes column?
        self.__max_fes: StatWriter | None = None
        #: do we put the max time millis column?
        self.__max_time_millis: StatWriter | None = None

    def setup(self, data: Iterable[EndStatistics]) -> "CsvWriter":
        """
        Set up this csv writer based on existing data.

        :param data: the data to setup with
        :returns: this writer
        """
        if self.__setup:
            raise ValueError(
                "EndStatistics CsvWriter has already been set up.")
        self.__setup = True

        data = reiterable(data)

        checker: int = 127
        for es in data:
            if es.algorithm is not None:
                self.__has_algorithm = True
                checker &= ~1
            if es.instance is not None:
                self.__has_instance = True
                checker &= ~2
            if es.objective is not None:
                self.__has_objective = True
                checker &= ~4
            if es.encoding is not None:
                self.__has_encoding = True
                checker &= ~8
            if es.n_success is not None:
                self.__has_n_success = True
                checker &= ~16
            if es.ert_fes is not None:
                self.__has_ert_fes = True
                checker &= ~32
            if es.ert_time_millis is not None:
                self.__has_ert_time_millis = True
                checker &= ~64
            if checker == 0:
                break

        scope: Final[str | None] = self.scope
        self.__goal_f = _to_csv_writer(
            EndStatistics.get_goal_f, EndStatistics.get_n,
            data, csv_scope(scope, KEY_GOAL_F),
            KEY_GOAL_F,
            "the goal objective value after which the runs can stop")
        self.__best_f_scaled = _to_csv_writer(
            EndStatistics.get_best_f_scaled, EndStatistics.get_n,
            data, csv_scope(scope, KEY_BEST_F_SCALED), KEY_BEST_F_SCALED,
            f"best objective value reached ({KEY_BEST_F}), divided by"
            f" the goal objective value ({KEY_GOAL_F})")
        self.__success_fes = _to_csv_writer(
            EndStatistics.get_success_fes, EndStatistics.get_n_success,
            data, csv_scope(scope, KEY_SUCCESS_FES), KEY_SUCCESS_FES,
            f"the FEs needed to reach {KEY_GOAL_F} for the successful runs")
        self.__success_time_millis = _to_csv_writer(
            EndStatistics.get_success_time_millis,
            EndStatistics.get_n_success, data, csv_scope(
                scope, KEY_SUCCESS_TIME_MILLIS), KEY_SUCCESS_TIME_MILLIS,
            f"the milliseconds needed to reach {KEY_GOAL_F} for the "
            "successful runs")
        self.__max_fes = _to_csv_writer(
            EndStatistics.get_max_fes, EndStatistics.get_n,
            data, csv_scope(scope, KEY_MAX_FES), KEY_MAX_FES,
            "the maximum number of FEs in the computational budget")
        self.__max_time_millis = _to_csv_writer(
            EndStatistics.get_max_time_millis, EndStatistics.get_n,
            data, csv_scope(scope, KEY_MAX_TIME_MILLIS), KEY_MAX_TIME_MILLIS,
            "the maximum milliseconds per run in the computational budget")
        self.__best_f.setup(map(EndStatistics.get_best_f, data))
        self.__life.setup(map(EndStatistics.get_last_improvement_fe, data))
        self.__lims.setup(map(
            EndStatistics.get_last_improvement_time_millis, data))
        self.__total_fes.setup(map(EndStatistics.get_total_fes, data))
        self.__total_ms.setup(map(EndStatistics.get_total_time_millis, data))

        return self

    def get_column_titles(self, dest: Callable[[str], None]) -> None:
        """
        Get the column titles.

        :param dest: the destination string consumer
        """
        p: Final[str] = self.scope
        if self.__has_algorithm:
            dest(csv_scope(p, KEY_ALGORITHM))
        if self.__has_instance:
            dest(csv_scope(p, KEY_INSTANCE))
        if self.__has_objective:
            dest(csv_scope(p, KEY_OBJECTIVE_FUNCTION))
        if self.__has_encoding:
            dest(csv_scope(p, KEY_ENCODING))
        dest(csv_scope(p, KEY_N))
        self.__best_f.get_column_titles(dest)
        self.__life.get_column_titles(dest)
        self.__lims.get_column_titles(dest)
        self.__total_fes.get_column_titles(dest)
        self.__total_ms.get_column_titles(dest)
        if self.__goal_f is not None:
            self.__goal_f.get_column_titles(dest)
        if self.__best_f_scaled is not None:
            self.__best_f_scaled.get_column_titles(dest)
        if self.__has_n_success:
            dest(csv_scope(p, KEY_N_SUCCESS))
        if self.__success_fes is not None:
            self.__success_fes.get_column_titles(dest)
        if self.__success_time_millis is not None:
            self.__success_time_millis.get_column_titles(dest)
        if self.__has_ert_fes:
            dest(csv_scope(p, KEY_ERT_FES))
        if self.__has_ert_time_millis:
            dest(csv_scope(p, KEY_ERT_TIME_MILLIS))
        if self.__max_fes is not None:
            self.__max_fes.get_column_titles(dest)
        if self.__max_time_millis is not None:
            self.__max_time_millis.get_column_titles(dest)

    def get_row(self, data: EndStatistics,
                dest: Callable[[str], None]) -> None:
        """
        Render a single end result record to a CSV row.

        :param data: the end result record
        :param dest: the string consumer
        """
        if self.__has_algorithm:
            dest("" if data.algorithm is None else data.algorithm)
        if self.__has_instance:
            dest("" if data.instance is None else data.instance)
        if self.__has_objective:
            dest("" if data.objective is None else data.objective)
        if self.__has_encoding:
            dest("" if data.encoding is None else data.encoding)
        dest(str(data.n))
        self.__best_f.get_row(data.best_f, dest)
        self.__life.get_row(data.last_improvement_fe, dest)
        self.__lims.get_row(data.last_improvement_time_millis, dest)
        self.__total_fes.get_row(data.total_fes, dest)
        self.__total_ms.get_row(data.total_time_millis, dest)
        if self.__goal_f is not None:
            self.__goal_f.get_optional_row(data.goal_f, dest, data.n)
        if self.__best_f_scaled is not None:
            self.__best_f_scaled.get_optional_row(
                data.best_f_scaled, dest, data.n)
        if self.__has_n_success:
            dest(str(data.n_success))
        if self.__success_fes is not None:
            self.__success_fes.get_optional_row(
                data.success_fes, dest, data.n_success)
        if self.__success_time_millis is not None:
            self.__success_time_millis.get_optional_row(
                data.success_time_millis, dest, data.n_success)
        if self.__has_ert_fes:
            dest(num_or_none_to_str(data.ert_fes))
        if self.__has_ert_time_millis:
            dest(num_or_none_to_str(data.ert_time_millis))
        if self.__max_fes is not None:
            self.__max_fes.get_optional_row(data.max_fes, dest, data.n)
        if self.__max_time_millis is not None:
            self.__max_time_millis.get_optional_row(
                data.max_time_millis, dest, data.n)

    def get_header_comments(self, dest: Callable[[str], None]) -> None:
        """
        Get any possible header comments.

        :param dest: the destination
        """
        dest("Experiment End Results Statistics")
        dest("See the description at the bottom of the file.")

    def get_footer_comments(self, dest: Callable[[str], None]) -> None:
        """
        Get any possible footer comments.

        :param dest: the destination
        """
        dest("")
        scope: Final[str | None] = self.scope

        dest("This file presents statistics gathered over multiple runs "
             "of optimization algorithms applied to problem instances.")
        if scope:
            dest("All end result statistics records start with prefix "
                 f"{scope}{SCOPE_SEPARATOR}.")
        if self.__has_algorithm:
            dest(f"{csv_scope(scope, KEY_ALGORITHM)}: {DESC_ALGORITHM}")
        if self.__has_instance:
            dest(f"{csv_scope(scope, KEY_INSTANCE)}: {DESC_INSTANCE}")
        if self.__has_objective:
            dest(f"{csv_scope(scope, KEY_OBJECTIVE_FUNCTION)}:"
                 f" {DESC_OBJECTIVE_FUNCTION}")
        if self.__has_encoding:
            dest(f"{csv_scope(scope, KEY_ENCODING)}: {DESC_ENCODING}")
        dest(f"{csv_scope(scope, KEY_N)}: the number of runs that were "
             f"performed for the given setup.")

        self.__best_f.get_footer_comments(dest)
        dest(f"In summary {csv_scope(scope, KEY_BEST_F)} is {DESC_BEST_F}.")

        self.__life.get_footer_comments(dest)
        dest(f"In summary {csv_scope(scope, KEY_LAST_IMPROVEMENT_FE)} "
             f"is {DESC_LAST_IMPROVEMENT_FE}.")

        self.__lims.get_footer_comments(dest)
        dest("In summary "
             f"{csv_scope(scope, KEY_LAST_IMPROVEMENT_TIME_MILLIS)} "
             f"is {DESC_LAST_IMPROVEMENT_TIME_MILLIS}.")

        self.__total_fes.get_footer_comments(dest)
        dest(f"In summary {csv_scope(scope, KEY_TOTAL_FES)} "
             f"is {DESC_TOTAL_FES}.")

        self.__total_ms.get_footer_comments(dest)
        dest(f"In summary {csv_scope(scope, KEY_TOTAL_TIME_MILLIS)} "
             f"is {DESC_TOTAL_TIME_MILLIS}.")

        if self.__goal_f is not None:
            self.__goal_f.get_footer_comments(dest)
            dest(f"In summary {csv_scope(scope, KEY_GOAL_F)} is"
                 f" {DESC_GOAL_F}.")

        if self.__best_f_scaled is not None:
            self.__best_f_scaled.get_footer_comments(dest)
            dest(f"In summary {csv_scope(scope, KEY_BEST_F_SCALED)} describes"
                 " the best objective value reached ("
                 f"{csv_scope(scope, KEY_BEST_F)}) divided by the goal "
                 f"objective value ({csv_scope(scope, KEY_GOAL_F)}).")

        if self.__has_n_success:
            dest(f"{csv_scope(scope, KEY_N_SUCCESS)} is the number of runs "
                 "that reached goal objective value "
                 f"{csv_scope(scope, KEY_GOAL_F)}. Obviously, "
                 f"0<={csv_scope(scope, KEY_N_SUCCESS)}<="
                 f"{csv_scope(scope, KEY_N)}.")
        if self.__success_fes is not None:
            self.__success_fes.get_footer_comments(dest)
            dest(f"{csv_scope(scope, KEY_SUCCESS_FES)} offers statistics "
                 "about the number of FEs that the 0<="
                 f"{csv_scope(scope, KEY_N_SUCCESS)}<="
                 f"{csv_scope(scope, KEY_N)} successful runs needed to reach "
                 f"the goal objective value {csv_scope(scope, KEY_GOAL_F)}.")

        if self.__success_time_millis is not None:
            self.__success_fes.get_footer_comments(dest)
            dest(f"{csv_scope(scope, KEY_SUCCESS_TIME_MILLIS)} offers "
                 "statistics about the number of milliseconds of clock time "
                 f"that the 0<={csv_scope(scope, KEY_N_SUCCESS)}<="
                 f"{csv_scope(scope, KEY_N)} successful runs needed to reach "
                 f"the goal objective value {csv_scope(scope, KEY_GOAL_F)}.")

        if self.__has_ert_fes:
            dest(f"{csv_scope(scope, KEY_ERT_FES)} is the empirical estimate"
                 " of the number of FEs to solve the problem. It can be "
                 "approximated by dividing the sum of "
                 f"{csv_scope(scope, KEY_TOTAL_FES)} over all runs by the "
                 f"number {csv_scope(scope, KEY_N_SUCCESS)} of successful "
                 "runs.")

        if self.__has_ert_time_millis:
            dest(f"{csv_scope(scope, KEY_ERT_TIME_MILLIS)} is the empirical "
                 "estimate of the number of FEs to solve the problem. It can "
                 "be approximated by dividing the sum of "
                 f"{csv_scope(scope, KEY_TOTAL_TIME_MILLIS)} over all runs by"
                 f" the number {csv_scope(scope, KEY_N_SUCCESS)} of "
                 "successful runs.")

        if self.__max_fes is not None:
            self.__max_fes.get_footer_comments(dest)
            dest(f"In summary {csv_scope(scope, KEY_MAX_FES)} is"
                 f" {DESC_MAX_FES}.")
        if self.__max_time_millis is not None:
            self.__max_time_millis.get_footer_comments(dest)
            dest(f"In summary {csv_scope(scope, KEY_MAX_TIME_MILLIS)} is"
                 f" {DESC_MAX_TIME_MILLIS}.")
        _csv_motipy_footer(dest)


class CsvReader:
    """A csv parser for end results."""

    def __init__(self, columns: dict[str, int]) -> None:
        """
        Create a CSV parser for :class:`EndResult`.

        :param columns: the columns
        """
        super().__init__()
        if not isinstance(columns, dict):
            raise type_error(columns, "columns", dict)

        #: the index of the algorithm column, if any
        self.__idx_algorithm: Final[int | None] = csv_column_or_none(
            columns, KEY_ALGORITHM)
        #: the index of the instance column, if any
        self.__idx_instance: Final[int | None] = csv_column_or_none(
            columns, KEY_INSTANCE)
        #: the index of the objective column, if any
        self.__idx_objective: Final[int | None] = csv_column_or_none(
            columns, KEY_OBJECTIVE_FUNCTION)
        #: the index of the encoding column, if any
        self.__idx_encoding: Final[int | None] = csv_column_or_none(
            columns, KEY_ENCODING)

        #: the index of the `N` column, i.e., where the number of runs is
        #: stored
        self.idx_n: Final[int] = csv_column(columns, KEY_N, True)

        n_key: Final[tuple[tuple[str, int]]] = ((KEY_N, self.idx_n), )
        #: the reader for the best-objective-value-reached statistics
        self.__best_f: Final[StatReader] = csv_select_scope(
            StatReader, columns, KEY_BEST_F, n_key)
        #: the reader for the last improvement FE statistics
        self.__life: Final[StatReader] = csv_select_scope(
            StatReader, columns, KEY_LAST_IMPROVEMENT_FE, n_key)
        #: the reader for the last improvement millisecond index statistics
        self.__lims: Final[StatReader] = csv_select_scope(
            StatReader, columns, KEY_LAST_IMPROVEMENT_TIME_MILLIS, n_key)
        #: the reader for the total FEs statistics
        self.__total_fes: Final[StatReader] = csv_select_scope(
            StatReader, columns, KEY_TOTAL_FES, n_key)
        #: the reader for the total milliseconds consumed statistics
        self.__total_ms: Final[StatReader] = csv_select_scope(
            StatReader, columns, KEY_TOTAL_TIME_MILLIS, n_key)

        #: the reader for the goal objective value statistics, if any
        self.__goal_f: Final[StatReader | None] = csv_select_scope_or_none(
            StatReader, columns, KEY_GOAL_F, n_key)
        #: the reader for the best-f / goal-f statistics, if any
        self.__best_f_scaled: Final[StatReader | None] = \
            csv_select_scope_or_none(
                StatReader, columns, KEY_BEST_F_SCALED, n_key)

        #: the index of the column where the number of successful runs is
        #: stored
        self.__idx_n_success: Final[int | None] = csv_column_or_none(
            columns, KEY_N_SUCCESS)
        succ_key: Final[tuple[tuple[str, int], ...]] = () \
            if self.__idx_n_success is None else (
            (KEY_N, self.__idx_n_success), )
        #: the reader for the success FE data, if any
        self.__success_fes: Final[StatReader | None] = \
            None if self.__idx_n_success is None else \
            csv_select_scope_or_none(
                StatReader, columns, KEY_SUCCESS_FES, succ_key)
        #: the reader for the success time milliseconds data, if any
        self.__success_time_millis: Final[StatReader | None] = \
            None if self.__idx_n_success is None else \
            csv_select_scope_or_none(
                StatReader, columns, KEY_SUCCESS_TIME_MILLIS, succ_key)

        #: the index of the expected FEs until success
        self.__idx_ert_fes: Final[int | None] = csv_column_or_none(
            columns, KEY_ERT_FES)
        #: the index of the expected milliseconds until success
        self.__idx_ert_time_millis: Final[int | None] = csv_column_or_none(
            columns, KEY_ERT_TIME_MILLIS)

        #: the columns with the maximum FE-based budget statistics
        self.__max_fes: Final[StatReader | None] = csv_select_scope_or_none(
            StatReader, columns, KEY_MAX_FES, n_key)
        #: the columns with the maximum time-based budget statistics
        self.__max_time_millis: Final[StatReader | None] = \
            csv_select_scope_or_none(
                StatReader, columns, KEY_MAX_TIME_MILLIS, n_key)

    def parse_row(self, data: list[str]) -> EndStatistics:
        """
        Parse a row of data.

        :param data: the data row
        :return: the end result statistics
        """
        return EndStatistics(
            algorithm=csv_str_or_none(data, self.__idx_algorithm),
            instance=csv_str_or_none(data, self.__idx_instance),
            objective=csv_str_or_none(data, self.__idx_objective),
            encoding=csv_str_or_none(data, self.__idx_encoding),
            n=int(data[self.idx_n]),
            best_f=self.__best_f.parse_row(data),
            last_improvement_fe=self.__life.parse_row(data),
            last_improvement_time_millis=self.__lims.parse_row(data),
            total_fes=self.__total_fes.parse_row(data),
            total_time_millis=self.__total_ms.parse_row(data),
            goal_f=StatReader.parse_optional_row(self.__goal_f, data),
            best_f_scaled=StatReader.parse_optional_row(
                self.__best_f_scaled, data),
            n_success=csv_val_or_none(data, self.__idx_n_success, int),
            success_fes=StatReader.parse_optional_row(
                self.__success_fes, data),
            success_time_millis=StatReader.parse_optional_row(
                self.__success_time_millis, data),
            ert_fes=csv_val_or_none(data, self.__idx_ert_fes, str_to_num),
            ert_time_millis=csv_val_or_none(
                data, self.__idx_ert_time_millis, str_to_num),
            max_fes=StatReader.parse_optional_row(self.__max_fes, data),
            max_time_millis=StatReader.parse_optional_row(
                self.__max_time_millis, data),
        )


# Run end-results to stat file if executed as script
if __name__ == "__main__":
    parser: Final[argparse.ArgumentParser] = moptipy_argparser(
        __file__, "Build an end-results statistics CSV file.",
        "This program creates a CSV file with basic statistics on the "
        "end-of-run state of experiments conducted with moptipy. It "
        "therefore either parses a directory structure with log files "
        "(if src identifies a directory) or a end results CSV file (if"
        " src identifies a file). In the former case, the directory "
        "will follow the form 'algorithm/instance/log_file' with one "
        "log file per run. In the latter case, it will be a file "
        "generated by the end_results.py tool of moptipy. The output "
        "of this tool is a CSV file where the columns are separated by"
        " ';' and the rows contain the statistics.")
    def_src: str = "./evaluation/end_results.txt"
    if not os.path.isfile(def_src):
        def_src = "./results"
    parser.add_argument(
        "source", nargs="?", default=def_src,
        help="either the directory with moptipy log files or the path to the "
             "end-results CSV file", type=Path)
    parser.add_argument(
        "dest", type=Path, nargs="?",
        default="./evaluation/end_statistics.txt",
        help="the path to the end results statistics CSV file to be created")
    parser.add_argument(
        "--join_algorithms",
        help="compute statistics over all algorithms, i.e., the statistics"
             " are not separated by algorithm but all algorithms are treated "
             "as one", action="store_true")
    parser.add_argument(
        "--join_instances",
        help="compute statistics over all instances, i.e., the statistics"
             " are not separated by instance but all instances are treated "
             "as one", action="store_true")
    parser.add_argument(
        "--join_objectives",
        help="compute statistics over all objective functions, i.e., the "
             "statistics are not separated by objective functions but all "
             "objectives functions are treated as one", action="store_true")
    parser.add_argument(
        "--join_encodings",
        help="compute statistics over all encodings, i.e., the statistics"
             " are not separated by encodings but all encodings are treated "
             "as one", action="store_true")
    args: Final[argparse.Namespace] = parser.parse_args()

    src_path: Final[Path] = args.source
    end_results: Final[list[EndResult]] = []
    if src_path.is_file():
        logger(f"{src_path!r} identifies as file, load as end-results csv")
        end_results_from_csv(src_path, end_results.append)
    else:
        logger(f"{src_path!r} identifies as directory, load it as log files")
        end_results_from_logs(src_path, end_results.append)

    end_stats: Final[list[EndStatistics]] = []
    from_end_results(
        source=end_results, consumer=end_stats.append,
        join_all_algorithms=args.join_algorithms,
        join_all_instances=args.join_instances,
        join_all_objectives=args.join_objectives,
        join_all_encodings=args.join_encodings)
    to_csv(end_stats, args.dest)
