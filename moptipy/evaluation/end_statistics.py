"""
Statistics aggregated over multiple instances of `EndResult`.

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
from typing import Any, Callable, Final, Iterable, Union

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
    F_NAME_RAW,
    F_NAME_SCALED,
    KEY_ENCODING,
    KEY_N,
    KEY_OBJECTIVE_FUNCTION,
    MultiRunData,
)
from moptipy.evaluation.end_results import EndResult
from moptipy.evaluation.statistics import (
    CSV_COLS,
    EMPTY_CSV_ROW,
    KEY_STDDEV,
    Statistics,
)
from moptipy.utils.console import logger
from moptipy.utils.help import argparser
from moptipy.utils.logger import CSV_SEPARATOR, SCOPE_SEPARATOR
from moptipy.utils.math import try_int, try_int_div
from moptipy.utils.path import Path
from moptipy.utils.strings import num_to_str, sanitize_name, str_to_intfloat
from moptipy.utils.types import check_int_range, type_error, type_name_of

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

#: the internal getters that can work directly
_GETTERS_0: Final[dict[str, Callable[["EndStatistics"],
                                     int | float | None]]] = {
    KEY_N_SUCCESS: lambda s: s.n_success,
    KEY_ERT_FES: lambda s: s.ert_fes,
    KEY_ERT_TIME_MILLIS: lambda s: s.ert_time_millis,
    KEY_GOAL_F: lambda s: s.goal_f if isinstance(s.goal_f,
                                                 int | float) else None,
    KEY_MAX_TIME_MILLIS: lambda s: s.max_time_millis
    if isinstance(s.max_time_millis, int | float) else None,
    KEY_MAX_FES: lambda s: s.max_fes
    if isinstance(s.max_fes, int | float) else None,
}

#: the internal getters that access end statistics
_GETTERS_1: Final[dict[
    str, Callable[["EndStatistics"], int | float | Statistics | None]]] = {
    KEY_LAST_IMPROVEMENT_FE: lambda s: s.last_improvement_fe,
    KEY_LAST_IMPROVEMENT_TIME_MILLIS:
        lambda s: s.last_improvement_time_millis,
    KEY_TOTAL_FES: lambda s: s.total_fes,
    KEY_TOTAL_TIME_MILLIS: lambda s: s.total_time_millis,
    F_NAME_RAW: lambda s: s.best_f,
    F_NAME_SCALED: lambda s: s.best_f_scaled,
    KEY_MAX_TIME_MILLIS: lambda s: s.max_time_millis,
    KEY_MAX_FES: lambda s: s.max_fes,
    KEY_GOAL_F: lambda s: s.goal_f,
}
_GETTERS_1[KEY_BEST_F] = _GETTERS_1[F_NAME_RAW]
_GETTERS_1[KEY_BEST_F_SCALED] = _GETTERS_1[F_NAME_SCALED]


@dataclass(frozen=True, init=False, order=False, eq=False)
class EndStatistics(MultiRunData):
    """
    Statistics over end results of one or multiple algorithm*instance setups.

    If one algorithm*instance is used, then `algorithm` and `instance` are
    defined. Otherwise, only the parameter which is the same over all recorded
    runs is defined.
    """

    #: The statistics about the best encountered result.
    best_f: Statistics
    #: The statistics about the last improvement FE.
    last_improvement_fe: Statistics
    #: The statistics about the last improvement time.
    last_improvement_time_millis: Statistics
    #: The statistics about the total number of FEs.
    total_fes: Statistics
    #: The statistics about the total time.
    total_time_millis: Statistics
    #: The goal objective value.
    goal_f: Statistics | int | float | None
    #: best_f / goal_f if goal_f is consistently defined and always positive.
    best_f_scaled: Statistics | None
    #: The number of successful runs, if goal_f != None, else None.
    n_success: int | None
    #: The FEs to success, if n_success > 0, None otherwise.
    success_fes: Statistics | None
    #: The time to success, if n_success > 0, None otherwise.
    success_time_millis: Statistics | None
    #: The ERT if FEs, while is inf if n_success=0, None if goal_f is None,
    #: and finite otherwise.
    ert_fes: int | float | None
    #: The ERT if milliseconds, while is inf if n_success=0, None if goal_f
    #: is None, and finite otherwise.
    ert_time_millis: int | float | None
    #: The budget in FEs, if every run had one; None otherwise.
    max_fes: Statistics | int | None
    #: The budget in milliseconds, if every run had one; None otherwise.
    max_time_millis: Statistics | int | None

    def __init__(self,
                 algorithm: str | None,
                 instance: str | None,
                 objective: str | None,
                 encoding: str | None,
                 n: int,
                 best_f: Statistics,
                 last_improvement_fe: Statistics,
                 last_improvement_time_millis: Statistics,
                 total_fes: Statistics,
                 total_time_millis: Statistics,
                 goal_f: float | int | Statistics | None,
                 best_f_scaled: Statistics | None,
                 n_success: int | None,
                 success_fes: Statistics | None,
                 success_time_millis: Statistics | None,
                 ert_fes: int | float | None,
                 ert_time_millis: int | float | None,
                 max_fes: Statistics | int | None,
                 max_time_millis: Statistics | int | None):
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
            this is the `Statistics` record about them
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

        if not isinstance(best_f, Statistics):
            raise type_error(best_f, "best_f", Statistics)
        object.__setattr__(self, "best_f", best_f)

        if not isinstance(last_improvement_fe, Statistics):
            raise type_error(last_improvement_fe, "last_improvement_fe",
                             Statistics)
        check_int_range(
            last_improvement_fe.minimum, "last_improvement_fe.minimum",
            1, 1_000_000_000_000_000)
        check_int_range(
            last_improvement_fe.maximum, "last_improvement_fe.maximum",
            last_improvement_fe.minimum, 1_000_000_000_000_000)
        object.__setattr__(self, "last_improvement_fe", last_improvement_fe)

        if not isinstance(last_improvement_time_millis, Statistics):
            raise type_error(last_improvement_time_millis,
                             "last_improvement_time_millis", Statistics)
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
        if not isinstance(total_fes, Statistics):
            raise type_error(total_fes, "total_fes", Statistics)
        check_int_range(
            total_fes.minimum, "total_fes.minimum",
            last_improvement_fe.minimum, 1_000_000_000_000_000)
        check_int_range(
            total_fes.maximum, "total_fes.maximum",
            max(total_fes.minimum, last_improvement_fe.maximum),
            1_000_000_000_000_000)
        object.__setattr__(self, "total_fes", total_fes)

        if not isinstance(total_time_millis, Statistics):
            raise type_error(total_time_millis, "total_time_millis",
                             Statistics)
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
            if isinstance(goal_f, float):
                goal_f = None if goal_f <= (-inf) else try_int(goal_f)
            elif not isinstance(goal_f, int | Statistics):
                raise type_error(goal_f, "goal_f", (int, Statistics))

            if best_f_scaled is not None:
                goal_f_min: Final[int | float] = \
                    goal_f.minimum if isinstance(goal_f, Statistics) \
                    else goal_f
                if goal_f_min <= 0:
                    raise ValueError(
                        f"best_f_scaled must be None if minimum goal_f "
                        f"({goal_f_min}) of goal_f {goal_f} is not positive,"
                        f" but is {best_f_scaled}.")
                if not isinstance(best_f_scaled, Statistics):
                    raise type_error(best_f_scaled, "best_f_scaled",
                                     Statistics)
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
                if not isinstance(success_fes, Statistics):
                    raise type_error(success_fes,
                                     "if n_success>0, then success_fes",
                                     Statistics)
                check_int_range(
                    success_fes.minimum, "success_fes.minimum",
                    last_improvement_fe.minimum, 1_000_000_000_000_000)
                check_int_range(
                    success_fes.maximum, "success_fes.maximum",
                    success_fes.minimum, last_improvement_fe.maximum)
                if not isinstance(success_time_millis, Statistics):
                    raise type_error(
                        success_time_millis,
                        "if n_success>0, then success_time_millis",
                        Statistics)
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

        if isinstance(max_fes, int):
            if max_fes < total_fes.maximum:
                raise ValueError(f"max_fes must be >= "
                                 f"{total_fes.maximum}, but is {max_fes}.")
        elif isinstance(max_fes, Statistics):
            if max_fes.minimum < total_fes.minimum:
                raise ValueError(
                    f"max_fes.minimum must be >= {total_fes.minimum},"
                    f" but is {max_fes.minimum}.")
            if max_fes.maximum < total_fes.maximum:
                raise ValueError(
                    f"max_fes.maximum must be >= {total_fes.maximum},"
                    f" but is {max_fes.maximum}.")
        elif max_fes is not None:
            raise type_error(max_fes, "max_fes", (int, Statistics, None))
        object.__setattr__(self, "max_fes", max_fes)

        if isinstance(max_time_millis, int):
            _check_max_time_millis(max_time_millis,
                                   total_fes.minimum,
                                   total_time_millis.maximum)
        elif isinstance(max_time_millis, Statistics):
            _check_max_time_millis(max_time_millis.minimum,
                                   total_fes.minimum,
                                   total_time_millis.minimum)
            _check_max_time_millis(max_time_millis.maximum,
                                   total_fes.minimum,
                                   total_time_millis.maximum)
        elif max_time_millis is not None:
            raise type_error(max_time_millis, "max_time_millis",
                             (int, Statistics, None))
        object.__setattr__(self, "max_time_millis", max_time_millis)

    @staticmethod
    def create(source: Iterable[EndResult]) -> "EndStatistics":
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
            Statistics.create(best_f),
            Statistics.create(last_improvement_fe),
            Statistics.create(last_improvement_time_millis),
            Statistics.create(total_fes),
            Statistics.create(total_time_millis),
            None if (goal_f is None)
            else (goal_f[0] if goal_f_same else Statistics.create(goal_f)),
            None if (best_f_scaled is None)
            else Statistics.create(best_f_scaled),
            n_success,
            None if (n_success is None) or (n_success <= 0)
            else Statistics.create(success_fes),
            None if (n_success is None) or (n_success <= 0)
            else Statistics.create(success_times),
            None if (n_success is None)
            else (inf if (n_success <= 0) else try_int_div(fes, n_success)),
            None if (n_success is None) else
            (inf if (n_success <= 0) else try_int_div(time, n_success)),
            None if max_fes is None else
            (max_fes[0] if max_fes_same else Statistics.create(max_fes)),
            None if max_time_millis is None
            else (max_time_millis[0] if max_time_same
                  else Statistics.create(max_time_millis)))

    @staticmethod
    def from_end_results(source: Iterable[EndResult],
                         consumer: Callable[["EndStatistics"], Any],
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
            consumer(EndStatistics.create(source))
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
                consumer(EndStatistics.create(sorter[key]))
        else:
            consumer(EndStatistics.create(
                next(iter(sorter.values()))))

    @staticmethod
    def to_csv(  # noqa
            data: Union["EndStatistics", Iterable["EndStatistics"]],  # noqa
            file: str) -> Path:  # noqa
        """
        Store a set of :class:`EndStatistics` in a CSV file.

        :param data: the data to store
        :param file: the file to generate
        :return: the path to the generated CSV file
        """
        path: Final[Path] = Path.path(file)
        logger(f"Writing end result statistics to CSV file {path!r}.")
        Path.path(os.path.dirname(path)).ensure_dir_exists()

        has_algorithm: bool = False  # 1
        has_instance: bool = False  # 2
        has_objective: bool = False  # 4
        has_encoding: bool = False  # 8
        has_goal_f: int = 0  # 16
        has_best_f_scaled: bool = False  # 32
        has_n_success: bool = False  # 64
        has_success_fes: bool = False  # 128
        has_success_time_millis: bool = False  # 256
        has_ert_fes: bool = False  # 512
        has_ert_time_millis: bool = False  # 1024
        has_max_fes: int = 0  # 2048
        has_max_time_millis: int = 0  # 4096
        checker: int = 8191

        if isinstance(data, EndStatistics):
            data = [data]

        for es in data:
            if es.algorithm is not None:
                has_algorithm = True
                checker &= ~1
            if es.instance is not None:
                has_instance = True
                checker &= ~2
            if es.objective is not None:
                has_objective = True
                checker &= ~4
            if es.encoding is not None:
                has_encoding = True
                checker &= ~8
            if es.goal_f is not None:
                if isinstance(es.goal_f, Statistics) \
                        and (es.goal_f.maximum > es.goal_f.minimum):
                    has_goal_f = 2
                    checker &= ~16
                elif has_goal_f == 0:
                    has_goal_f = 1
            if es.best_f_scaled is not None:
                has_best_f_scaled = True
                checker &= ~32
            if es.n_success is not None:
                has_n_success = True
                checker &= ~64
            if es.success_fes is not None:
                has_success_fes = True
                checker &= ~128
            if es.success_time_millis is not None:
                has_success_time_millis = True
                checker &= ~256
            if es.ert_fes is not None:
                has_ert_fes = True
                checker &= ~512
            if es.ert_time_millis is not None:
                has_ert_time_millis = True
                checker &= ~1024
            if es.max_fes is not None:
                if isinstance(es.max_fes, Statistics) \
                        and (es.max_fes.maximum > es.max_fes.minimum):
                    has_max_fes = 2
                    checker &= ~2048
                elif has_max_fes == 0:
                    has_max_fes = 1
            if es.max_time_millis is not None:
                if isinstance(es.max_time_millis, Statistics) \
                        and (es.max_time_millis.maximum
                             > es.max_time_millis.minimum):
                    has_max_time_millis = 2
                    checker &= ~4096
                elif has_max_time_millis == 0:
                    has_max_time_millis = 1
            if checker == 0:
                break

        with path.open_for_write() as out:
            wrt: Final[Callable] = out.write
            sep: Final[str] = CSV_SEPARATOR
            if has_algorithm:
                wrt(KEY_ALGORITHM)
                wrt(sep)
            if has_instance:
                wrt(KEY_INSTANCE)
                wrt(sep)
            if has_objective:
                wrt(KEY_OBJECTIVE_FUNCTION)
                wrt(sep)
            if has_encoding:
                wrt(KEY_ENCODING)
                wrt(sep)

            def h(p) -> None:
                wrt(sep.join(Statistics.csv_col_names(p)))

            wrt(KEY_N)
            wrt(sep)
            h(KEY_BEST_F)
            wrt(sep)
            h(KEY_LAST_IMPROVEMENT_FE)
            wrt(sep)
            h(KEY_LAST_IMPROVEMENT_TIME_MILLIS)
            wrt(sep)
            h(KEY_TOTAL_FES)
            wrt(sep)
            h(KEY_TOTAL_TIME_MILLIS)
            if has_goal_f == 1:
                wrt(sep)
                wrt(KEY_GOAL_F)
            elif has_goal_f == 2:
                wrt(sep)
                h(KEY_GOAL_F)
            if has_best_f_scaled:
                wrt(sep)
                h(KEY_BEST_F_SCALED)
            if has_n_success:
                wrt(sep)
                wrt(KEY_N_SUCCESS)
            if has_success_fes:
                wrt(sep)
                h(KEY_SUCCESS_FES)
            if has_success_time_millis:
                wrt(sep)
                h(KEY_SUCCESS_TIME_MILLIS)
            if has_ert_fes:
                wrt(sep)
                wrt(KEY_ERT_FES)
            if has_ert_time_millis:
                wrt(sep)
                wrt(KEY_ERT_TIME_MILLIS)
            if has_max_fes == 1:
                wrt(sep)
                wrt(KEY_MAX_FES)
            elif has_max_fes == 2:
                wrt(sep)
                h(KEY_MAX_FES)
            if has_max_time_millis == 1:
                wrt(sep)
                wrt(KEY_MAX_TIME_MILLIS)
            elif has_max_time_millis == 2:
                wrt(sep)
                h(KEY_MAX_TIME_MILLIS)
            out.write("\n")

            csv: Final[Callable] = Statistics.value_to_csv
            num: Final[Callable] = num_to_str

            for er in data:
                if has_algorithm:
                    if er.algorithm is not None:
                        wrt(er.algorithm)
                    wrt(sep)
                if has_instance:
                    if er.instance is not None:
                        wrt(er.instance)
                    wrt(sep)
                if has_objective:
                    if er.objective is not None:
                        wrt(er.objective)
                    wrt(sep)
                if has_encoding:
                    if er.encoding is not None:
                        wrt(er.encoding)
                    wrt(sep)
                wrt(str(er.n))
                wrt(sep)
                wrt(er.best_f.to_csv())
                wrt(sep)
                wrt(er.last_improvement_fe.to_csv())
                wrt(sep)
                wrt(er.last_improvement_time_millis.to_csv())
                wrt(sep)
                wrt(er.total_fes.to_csv())
                wrt(sep)
                wrt(er.total_time_millis.to_csv())
                if has_goal_f == 1:
                    wrt(sep)
                    if er.goal_f is not None:
                        wrt(num(er.goal_f.median if isinstance(
                            er.goal_f, Statistics) else er.goal_f))
                elif has_goal_f == 2:
                    wrt(sep)
                    if isinstance(er.goal_f, Statistics):
                        wrt(er.goal_f.to_csv())
                    elif isinstance(er.goal_f, int | float):
                        wrt(csv(er.goal_f))
                    else:
                        wrt(EMPTY_CSV_ROW)
                if has_best_f_scaled:
                    wrt(sep)
                    if er.best_f_scaled is None:
                        wrt(EMPTY_CSV_ROW)
                    else:
                        wrt(er.best_f_scaled.to_csv())
                if has_n_success:
                    wrt(sep)
                    if er.n_success is not None:
                        wrt(str(er.n_success))
                if has_success_fes:
                    wrt(sep)
                    if er.success_fes is None:
                        wrt(EMPTY_CSV_ROW)
                    else:
                        wrt(er.success_fes.to_csv())
                if has_success_time_millis:
                    wrt(sep)
                    if er.success_time_millis is None:
                        wrt(EMPTY_CSV_ROW)
                    else:
                        wrt(er.success_time_millis.to_csv())
                if has_ert_fes:
                    wrt(sep)
                    if er.ert_fes is not None:
                        wrt(num(er.ert_fes))
                if has_ert_time_millis:
                    wrt(sep)
                    if er.ert_time_millis is not None:
                        wrt(num(er.ert_time_millis))
                if has_max_fes == 1:
                    wrt(sep)
                    if er.max_fes is not None:
                        wrt(str(er.max_fes.median if isinstance(
                            er.max_fes, Statistics) else er.max_fes))
                elif has_max_fes == 2:
                    wrt(sep)
                    if isinstance(er.max_fes, Statistics):
                        wrt(er.max_fes.to_csv())
                    elif isinstance(er.max_fes, int | float):
                        wrt(csv(er.max_fes))
                    else:
                        wrt(EMPTY_CSV_ROW)
                if has_max_time_millis == 1:
                    wrt(sep)
                    if er.max_time_millis is not None:
                        wrt(str(er.max_time_millis.median if isinstance(
                            er.max_time_millis, Statistics)
                            else er.max_time_millis))
                elif has_max_time_millis == 2:
                    wrt(sep)
                    if isinstance(er.max_time_millis, Statistics):
                        wrt(er.max_time_millis.to_csv())
                    elif isinstance(er.max_time_millis, int | float):
                        wrt(csv(er.max_time_millis))
                    else:
                        wrt(EMPTY_CSV_ROW)
                out.write("\n")

        logger(f"Done writing end result statistics to CSV file {path!r}.")
        path.enforce_file()
        return path

    @staticmethod
    def from_csv(file: str,
                 consumer: Callable[["EndStatistics"], Any]) -> None:
        """
        Parse a CSV file and collect all encountered :class:`EndStatistics`.

        :param file: the file to parse
        :param consumer: the consumer to receive all the parsed instances of
            :class:`~moptipy.evaluation.end_statistics.EndStatistics`, can be
            the `append` method of a :class:`list`
        """
        path: Final[Path] = Path.file(file)
        logger(f"Begin reading end result statistics from CSV file {path!r}.")

        sep: Final[str] = CSV_SEPARATOR
        with path.open_for_read() as rd:
            headerstr: Final[str] = rd.readline()
            if not isinstance(headerstr, str):
                raise type_error(headerstr, f"{file!r}[0]", str)
            header: Final[list[str]] = [ss.strip()
                                        for ss in headerstr.strip().split(sep)]
            if len(header) <= 3:
                raise ValueError(
                    f"Invalid header {headerstr!r} in file {file!r}.")

            idx = 0
            has_algorithm: bool = False
            if header[0] == KEY_ALGORITHM:
                has_algorithm = True
                idx = 1

            has_instance: bool = False
            if header[idx] == KEY_INSTANCE:
                has_instance = True
                idx += 1

            has_objective: bool = False
            if header[idx] == KEY_OBJECTIVE_FUNCTION:
                has_objective = True
                idx += 1

            has_encoding: bool = False
            if header[idx] == KEY_ENCODING:
                has_encoding = True
                idx += 1

            csv: Final[Callable] = Statistics.csv_col_names

            if header[idx] != KEY_N:
                raise ValueError(
                    f"Expected to find {KEY_N!r} at index {idx} "
                    f"in header {headerstr!r} of file {path!r}.")
            idx += 1

            for key in [KEY_BEST_F, KEY_LAST_IMPROVEMENT_FE,
                        KEY_LAST_IMPROVEMENT_TIME_MILLIS,
                        KEY_TOTAL_FES, KEY_TOTAL_TIME_MILLIS]:
                if csv(key) != header[idx:(idx + CSV_COLS)]:
                    raise ValueError(
                        f"Expected to find '{key}.*' keys from index "
                        f"{idx} on in header "
                        f"{headerstr!r} of file {path!r}, expected "
                        f"{csv(key)} but got {header[idx:(idx + CSV_COLS)]}.")
                idx += CSV_COLS

            has_goal_f: int = 0
            has_best_f_scaled: bool = False
            has_n_success: bool = False
            has_success_fes: bool = False
            has_success_time: bool = False
            has_ert_fes: bool = False
            has_ert_time: bool = False
            has_max_fes: int = 0
            has_max_time: int = 0
            while idx <= len(header):
                if header[idx] == KEY_GOAL_F:
                    has_goal_f = 1
                    idx += 1
                elif header[idx].startswith(KEY_GOAL_F):
                    has_goal_f = 2
                    if csv(KEY_GOAL_F) != header[idx:(idx + CSV_COLS)]:
                        raise ValueError(
                            f"Expected to find '{KEY_GOAL_F}.*' keys from "
                            f"index {idx} on in header "
                            f"{headerstr!r} of file {path!r}.")
                    idx += CSV_COLS

                if idx >= len(header):
                    break

                if header[idx].startswith(KEY_BEST_F_SCALED):
                    has_best_f_scaled = True
                    if csv(KEY_BEST_F_SCALED) != \
                            header[idx:(idx + CSV_COLS)]:
                        raise ValueError(
                            f"Expected to find '{KEY_BEST_F_SCALED}.*' "
                            f"keys from index {idx} on in header "
                            f"{headerstr!r} of file {path!r}.")
                    idx += CSV_COLS

                if idx >= len(header):
                    break

                if header[idx] == KEY_N_SUCCESS:
                    has_n_success = True
                    idx += 1

                if idx >= len(header):
                    break

                if header[idx].startswith(KEY_SUCCESS_FES):
                    has_success_fes = True
                    if csv(KEY_SUCCESS_FES) != header[idx:(idx + CSV_COLS)]:
                        raise ValueError(
                            f"Expected to find '{KEY_SUCCESS_FES}.*' "
                            f"keys from index {idx} on in header "
                            f"{headerstr!r} of file {path!r}.")
                    idx += CSV_COLS

                if idx >= len(header):
                    break

                if header[idx].startswith(KEY_SUCCESS_TIME_MILLIS):
                    has_success_time = True
                    if csv(KEY_SUCCESS_TIME_MILLIS) != \
                            header[idx:(idx + CSV_COLS)]:
                        raise ValueError(
                            f"Expected to find '{KEY_SUCCESS_TIME_MILLIS}.*' "
                            f"keys from index {idx} on in header "
                            f"{headerstr!r} of file {path!r}.")
                    idx += CSV_COLS

                if idx >= len(header):
                    break

                if header[idx] == KEY_ERT_FES:
                    has_ert_fes = True
                    idx += 1

                if idx >= len(header):
                    break

                if header[idx] == KEY_ERT_TIME_MILLIS:
                    has_ert_time = True
                    idx += 1

                if idx >= len(header):
                    break

                if header[idx] == KEY_MAX_FES:
                    has_max_fes = 1
                    idx += 1
                elif header[idx].startswith(KEY_MAX_FES):
                    has_max_fes = 2
                    if csv(KEY_MAX_FES) != header[idx:(idx + CSV_COLS)]:
                        raise ValueError(
                            f"Expected to find '{KEY_MAX_FES}.*' keys"
                            f" from index {idx} on in header "
                            f"{headerstr!r} of file {path!r}.")
                    idx += CSV_COLS

                if idx >= len(header):
                    break

                if header[idx] == KEY_MAX_TIME_MILLIS:
                    has_max_time = 1
                    idx += 1
                elif header[idx].startswith(KEY_MAX_TIME_MILLIS):
                    has_max_time = 2
                    if csv(KEY_MAX_FES) != header[idx:(idx + CSV_COLS)]:
                        raise ValueError(
                            f"Expected to find '{KEY_MAX_TIME_MILLIS}.*' "
                            f"keys from index {idx} on in header "
                            f"{headerstr!r} of file {path!r}.")
                    idx += CSV_COLS

                break

            if len(header) > idx:
                raise ValueError(
                    f"Unexpected item {header[idx]!r} in header "
                    f"{header!r} of file {path!r}.")

            while True:
                lines = rd.readlines(100)
                if (lines is None) or (len(lines) <= 0):
                    break
                for line in lines:
                    row = [ss.strip() for ss in line.strip().split(sep)]

                    idx = 0
                    algorithm: str | None = None
                    instance: str | None = None
                    objective: str | None = None
                    encoding: str | None = None
                    n: int
                    goal_f: None | int | float | Statistics = None
                    best_f_scaled: Statistics | None = None
                    n_success: int | None = None
                    success_fes: Statistics | None = None
                    success_time: Statistics | None = None
                    ert_fes: int | float | None = None
                    ert_time: int | float | None = None
                    max_fes: int | Statistics | None = None
                    max_time: int | Statistics | None = None

                    try:
                        if has_algorithm:
                            algorithm = sanitize_name(row[0])
                            idx += 1

                        if has_instance:
                            instance = sanitize_name(row[idx])
                            idx += 1

                        if has_objective:
                            objective = sanitize_name(row[idx])
                            idx += 1

                        if has_encoding:
                            encoding = sanitize_name(row[idx])
                            idx += 1

                        n = int(row[idx])
                        idx += 1

                        best_f: Statistics = Statistics.from_csv(
                            n, row[idx:(idx + CSV_COLS)])
                        idx += CSV_COLS

                        last_improv_fe: Statistics = Statistics.from_csv(
                            n, row[idx:(idx + CSV_COLS)])
                        idx += CSV_COLS

                        last_improv_time: Statistics = Statistics.from_csv(
                            n, row[idx:(idx + CSV_COLS)])
                        idx += CSV_COLS

                        total_fes: Statistics = Statistics.from_csv(
                            n, row[idx:(idx + CSV_COLS)])
                        idx += CSV_COLS

                        total_time: Statistics = Statistics.from_csv(
                            n, row[idx:(idx + CSV_COLS)])
                        idx += CSV_COLS

                        if has_goal_f == 1:
                            goal_f = str_to_intfloat(row[idx])
                            idx += 1
                        elif has_goal_f == 2:
                            goal_f = Statistics.from_csv(
                                n, row[idx:(idx + CSV_COLS)])
                            idx += CSV_COLS

                        if has_best_f_scaled:
                            best_f_scaled = Statistics.from_csv(
                                n, row[idx:(idx + CSV_COLS)])
                            idx += CSV_COLS

                        if has_n_success:
                            n_success = int(row[idx])
                            idx += 1

                        if has_success_fes:
                            success_fes = Statistics.from_csv(
                                n_success, row[idx:(idx + CSV_COLS)]) \
                                if n_success > 0 else None
                            idx += CSV_COLS

                        if has_success_time:
                            success_time = Statistics.from_csv(
                                n_success, row[idx:(idx + CSV_COLS)]) \
                                if n_success > 0 else None
                            idx += CSV_COLS

                        if has_ert_fes:
                            ert_fes = str_to_intfloat(row[idx]) \
                                if n_success > 0 else inf
                            idx += 1

                        if has_ert_time:
                            ert_time = str_to_intfloat(row[idx]) \
                                if n_success > 0 else inf
                            idx += 1

                        if has_max_fes == 1:
                            max_fes = int(row[idx])
                            idx += 1
                        elif has_max_fes == 2:
                            max_fes = Statistics.from_csv(
                                n, row[idx:(idx + CSV_COLS)])
                            idx += CSV_COLS

                        if has_max_time == 1:
                            max_time = int(row[idx])
                            idx += 1
                        elif has_max_time == 2:
                            max_time = Statistics.from_csv(
                                n, row[idx:(idx + CSV_COLS)])
                            idx += CSV_COLS

                    except Exception as be:
                        raise ValueError(
                            f"Invalid row {line!r} in file {path!r}.") from be

                    if len(row) != idx:
                        raise ValueError("Invalid number of columns in row "
                                         f"{line!r} in file {path!r}.")
                    consumer(EndStatistics(
                        algorithm, instance, objective, encoding, n, best_f,
                        last_improv_fe, last_improv_time, total_fes,
                        total_time, goal_f, best_f_scaled, n_success,
                        success_fes, success_time, ert_fes, ert_time, max_fes,
                        max_time))

        logger("Finished reading end result statistics from CSV "
               f"file {path!r}.")

    @staticmethod
    def getter(dimension: str) -> Callable[["EndStatistics"],
                                           int | float | None]:
        """
        Create a function that obtains the given dimension from EndStatistics.

        :param dimension: the dimension
        :returns: a callable that returns the value corresponding to the
            dimension
        """
        if not isinstance(dimension, str):
            raise type_error(dimension, "dimension", str)
        if dimension in _GETTERS_0:
            return _GETTERS_0[dimension]

        ssi: int = dimension.find(SCOPE_SEPARATOR)
        if ssi <= 0:
            raise ValueError(f"unknown dimension {dimension!r}.")
        scope: str = dimension[:ssi]
        dim: str = dimension[ssi + 1:]
        if (len(scope) <= 0) or (len(dim) <= 0):
            raise ValueError(
                f"invalid dimension {dimension!r}, has "
                f"scope {scope!r} and sub-dimension {dim!r}")

        if scope not in _GETTERS_1:
            raise ValueError(
                f"invalid dimension {dimension!r}, has "
                f"unknown scope {scope!r} and sub-dimension {dim!r}")

        l1 = _GETTERS_1[scope]

        if dim != KEY_STDDEV:
            l2 = Statistics.getter(dim)

            def __inner_sat(s: EndStatistics, ll1=l1, ll2=l2) \
                    -> int | float | None:
                a: Final[Statistics] = ll1(s)
                if a is None:
                    return None
                if isinstance(a, int | float):
                    return a  # max, min, med, mean = a
                return ll2(a)  # apply statistics getter
            return __inner_sat

        def __inner_sd(s: EndStatistics, ll1=l1) -> int | float | None:
            a: Final[Statistics] = ll1(s)
            if a is None:
                return None
            if isinstance(a, int | float):
                return 0  # sd of a single number = 0
            return a.stddev
        return __inner_sd


# Run end-results to stat file if executed as script
if __name__ == "__main__":
    parser: Final[argparse.ArgumentParser] = argparser(
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
             "end-results CSV file", type=Path.path)
    parser.add_argument(
        "dest", type=Path.path, nargs="?",
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
        EndResult.from_csv(src_path, end_results.append)
    else:
        logger(f"{src_path!r} identifies as directory, load it as log files")
        EndResult.from_logs(src_path, end_results.append)

    end_stats: Final[list[EndStatistics]] = []
    EndStatistics.from_end_results(
        source=end_results, consumer=end_stats.append,
        join_all_algorithms=args.join_algorithms,
        join_all_instances=args.join_instances,
        join_all_objectives=args.join_objectives,
        join_all_encodings=args.join_encodings)
    EndStatistics.to_csv(end_stats, args.dest)
