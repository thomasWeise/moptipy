"""Statistics aggregated over multiple instances of :class:`EndResult`."""
from dataclasses import dataclass
from datetime import datetime
from math import inf
from typing import Optional, Union, Iterable, List, MutableSequence, Dict, \
    Final, Callable

from moptipy.evaluation._utils import _try_int, _try_div, _str_to_if, \
    _check_max_time_millis
from moptipy.evaluation.base import MultiRunData, KEY_N
from moptipy.evaluation.end_results import EndResult
from moptipy.evaluation.statistics import Statistics, EMPTY_CSV_ROW, CSV_COLS
from moptipy.utils import logging as log
from moptipy.utils.io import canonicalize_path, enforce_file

#: The key for the best F.
KEY_BEST_F_SCALED: Final[str] = log.KEY_BEST_F + "scaled"
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


@dataclass(frozen=True, init=False, order=True)
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
    goal_f: Union[Statistics, int, float, None]
    #: best_f / goal_f if goal_f is consistently defined and always positive.
    best_f_scaled: Optional[Statistics]
    #: The number of successful runs, if goal_f != None, else None.
    n_success: Optional[int]
    #: The FEs to success, if n_success > 0, None otherwise.
    success_fes: Optional[Statistics]
    #: The time to success, if n_success > 0, None otherwise.
    success_time_millis: Optional[Statistics]
    #: The ERT if FEs, while is inf if n_success=0, None if goal_f is None,
    #: and finite otherwise.
    ert_fes: Union[int, float, None]
    #: The ERT if milliseconds, while is inf if n_success=0, None if goal_f
    #: is None, and finite otherwise.
    ert_time_millis: Union[int, float, None]
    #: The budget in FEs, if every run had one; None otherwise.
    max_fes: Union[Statistics, int, None]
    #: The budget in milliseconds, if every run had one; None otherwise.
    max_time_millis: Union[Statistics, int, None]

    def __init__(self,
                 algorithm: Optional[str],
                 instance: Optional[str],
                 n: int,
                 best_f: Statistics,
                 last_improvement_fe: Statistics,
                 last_improvement_time_millis: Statistics,
                 total_fes: Statistics,
                 total_time_millis: Statistics,
                 goal_f: Union[float, int, Statistics, None],
                 best_f_scaled: Optional[Statistics],
                 n_success: Optional[int],
                 success_fes: Optional[Statistics],
                 success_time_millis: Optional[Statistics],
                 ert_fes: Union[int, float, None],
                 ert_time_millis: Union[int, float, None],
                 max_fes: Union[Statistics, int, None],
                 max_time_millis: Union[Statistics, int, None]):
        """
        Create the end statistics of an experiment-setup combination.

        :param Optional[str] algorithm: the algorithm name, if all runs are
            with the same algorithm
        :param Optional[str] instance: the instance name, if all runs are
            on the same instance
        :param int n: the total number of runs
        :param Statistics best_f: statistics about the best achieved result
        :param Statistics last_improvement_fe: statistics about the last
            improvement FE
        :param Statistics last_improvement_time_millis: statistics about the
            last improvement time
        :param Statistics total_fes: statistics about the total FEs
        :param Statistics total_time_millis: statistics about the total
            runtime in milliseconds
        :param Union[Statistics, int, float, None] goal_f: if the goal
            objective value is not defined sometimes, this will be `None`.
            If it is always defined and always the same, then this will be
            that value. If different goal values exist, then this is the
            `Statistics` record about them
        :param Optional[Statistics] best_f_scaled: if `goal_f` is not `None`,
            then here we provide statistics about `best_f` divided by the
            corresponding `goal_f`
        :param Optional[int] n_success: the number of successful runs is only
            defined if `goal_f` is not `None` and counts the number of runs
            that reach or surpass their corresponding `goal_f`
        :param Optional[Statistics] success_fes: if `goal_f` is not `None`,
            then this holds statistics about the last improvement FE of only
            the successful runs
        :param Optional[Statistics] success_time_millis: if `goal_f` is not
            `None`, then this holds statistics about the last improvement
            times of only the successful runs
        :param Union[int, float, None] ert_fes: if `goal_f` is always defined,
            then this is the empirically estimated running time to solve the
            problem in FEs if `n_success>0` and `inf` otherwise
        :param Union[int, float, None] ert_time_millis: if `goal_f` is always
            defined, then this is the empirically estimated running time to
            solve the problem in milliseconds if `n_success>0` and `inf`
            otherwise
        :param Union[Statistics, int, None] max_fes: the budget in FEs, if any
        :param Union[Statistics, int, None] max_time_millis: the budget in
            term of milliseconds
        """
        super().__init__(algorithm, instance, n)

        if not isinstance(best_f, Statistics):
            raise TypeError(f"best_f must Statistics, but is {type(best_f)}.")
        object.__setattr__(self, "best_f", best_f)

        if not isinstance(last_improvement_fe, Statistics):
            raise TypeError("last_improvement_fe must Statistics, "
                            f"but is {type(last_improvement_fe)}.")
        if last_improvement_fe.minimum <= 0:
            raise ValueError("No last_improvement_fe can be <= 0, but "
                             f"encountered {last_improvement_fe.minimum}.")
        if not isinstance(last_improvement_fe.minimum, int):
            raise ValueError(
                "Minimum last_improvement_fe must be int, but encountered "
                f"{type(last_improvement_fe.minimum)}.")
        if not isinstance(last_improvement_fe.maximum, int):
            raise ValueError(
                "Maximum last_improvement_fe must be int, but encountered "
                f"{type(last_improvement_fe.maximum)}.")
        object.__setattr__(self, "last_improvement_fe", last_improvement_fe)

        if not isinstance(last_improvement_time_millis, Statistics):
            raise TypeError("last_improvement_time_millis must Statistics, "
                            f"but is {type(last_improvement_time_millis)}.")
        object.__setattr__(self, "last_improvement_time_millis",
                           last_improvement_time_millis)
        if last_improvement_time_millis.minimum < 0:
            raise ValueError(
                "No last_improvement_time_millis can be < 0, but encountered "
                f"{last_improvement_time_millis.minimum}.")
        if not isinstance(last_improvement_time_millis.minimum, int):
            raise ValueError(
                "Minimum last_improvement_time_millis must be int, but "
                f"encountered {type(last_improvement_time_millis.minimum)}.")
        if not isinstance(last_improvement_time_millis.maximum, int):
            raise ValueError(
                "Maximum last_improvement_time_millis must be int, but "
                f"encountered {type(last_improvement_time_millis.maximum)}.")

        if not isinstance(total_fes, Statistics):
            raise TypeError("total_fes must Statistics, "
                            f"but is {type(total_fes)}.")
        if total_fes.minimum <= 0:
            raise ValueError("No total_fes can be <= 0, but "
                             f"encountered {total_fes.minimum}.")
        if not isinstance(total_fes.minimum, int):
            raise ValueError(
                "Minimum total_fes must be int, but encountered "
                f"{type(total_fes.minimum)}.")
        if not isinstance(total_fes.maximum, int):
            raise ValueError(
                "Maximum total_fes must be int, but encountered "
                f"{type(total_fes.maximum)}.")
        if total_fes.minimum < last_improvement_fe.minimum:
            raise ValueError(
                f"Minimum total_fes ({total_fes.minimum}) cannot be"
                "less than minimum last_improvement_fe "
                f"({last_improvement_fe.minimum}).")
        if total_fes.maximum < last_improvement_fe.maximum:
            raise ValueError(
                f"Maximum total_fes ({total_fes.maximum}) cannot be"
                "less than maximum last_improvement_fe "
                f"({last_improvement_fe.maximum}).")
        object.__setattr__(self, "total_fes", total_fes)

        if not isinstance(total_time_millis, Statistics):
            raise TypeError("total_time_millis must Statistics, "
                            f"but is {type(total_time_millis)}.")
        if total_time_millis.minimum < 0:
            raise ValueError("No total_time_millis can be < 0, but "
                             f"encountered {total_time_millis.minimum}.")
        if not isinstance(total_time_millis.minimum, int):
            raise ValueError(
                "Minimum total_time_millis must be int, but encountered "
                f"{type(total_time_millis.minimum)}.")
        if not isinstance(total_time_millis.maximum, int):
            raise ValueError(
                "Maximum total_time_millis must be int, but encountered "
                f"{type(total_time_millis.maximum)}.")
        if total_time_millis.minimum < last_improvement_time_millis.minimum:
            raise ValueError(
                f"Minimum total_time_millis ({total_time_millis.minimum}) "
                "cannot be less than minimum last_improvement_time_millis "
                f"({last_improvement_time_millis.minimum}).")
        if total_time_millis.maximum < last_improvement_time_millis.maximum:
            raise ValueError(
                f"Maximum total_time_millis ({total_time_millis.maximum}) "
                "cannot be less than maximum last_improvement_time_millis "
                f"({last_improvement_time_millis.maximum}).")
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
                if goal_f <= (-inf):
                    goal_f = None
                else:
                    goal_f = _try_int(goal_f)
            elif not isinstance(goal_f, (int, Statistics)):
                raise TypeError("goal_f must be int, float, None, or "
                                f"Statistics, but is {type(goal_f)}.")

            if best_f_scaled is not None:
                if not isinstance(best_f_scaled, Statistics):
                    raise TypeError("best_f_scaled must be Statistics, "
                                    f"but is {type(best_f_scaled)}.")
                if best_f_scaled.minimum < 0:
                    raise ValueError(
                        "best_f_scaled cannot be negative, but encountered "
                        f"{best_f_scaled.minimum}.")

            if not isinstance(n_success, int):
                raise TypeError("n_success must be int, "
                                f"but is {type(n_success)}.")
            if n_success < 0:
                raise ValueError(
                    f"n_success must be positive, but is {n_success}.")

            if not isinstance(ert_fes, float):
                raise TypeError(
                    f"ert_fes must be float, but it's a {type(ert_fes)}.")
            if not isinstance(ert_time_millis, float):
                raise TypeError("ert_time_millis must be float, "
                                f"but it's a {type(ert_time_millis)}.")

            if n_success > 0:
                if not isinstance(success_fes, Statistics):
                    raise ValueError(
                        "If n_success>0, then success_fes must be"
                        f" Statistics, but it's a {type(success_fes)}.")
                if success_fes.minimum < last_improvement_fe.minimum:
                    raise ValueError(
                        "success_fes.minimum must be >= "
                        f"{last_improvement_fe.minimum}, but is "
                        f"{success_fes.minimum}.")
                if success_fes.maximum > last_improvement_fe.maximum:
                    raise ValueError(
                        "success_fes.maximum must be <= "
                        f"{last_improvement_fe.maximum}, but is "
                        f"{success_fes.maximum}.")
                if not isinstance(success_time_millis, Statistics):
                    raise ValueError(
                        "If n_success>0, then success_time_millis must be"
                        " Statistics, but it's a"
                        f" {type(success_time_millis)}.")
                if success_time_millis.minimum < \
                        last_improvement_time_millis.minimum:
                    raise ValueError(
                        "success_time_millis.minimum must be >= "
                        f"{last_improvement_time_millis.minimum}, but is "
                        f"{success_time_millis.minimum}.")
                if success_time_millis.maximum > \
                        last_improvement_time_millis.maximum:
                    raise ValueError(
                        "success_time_millis.maximum must be <= "
                        f"{last_improvement_time_millis.maximum}, but is "
                        f"{success_time_millis.maximum}.")

                ert_fes = _try_int(ert_fes)
                if ert_fes < success_fes.minimum:
                    raise ValueError(
                        "ert_fes must be >= "
                        f"{success_fes.minimum}, but is {ert_fes}.")
                ert_fe_max = total_fes.mean_arith * n
                if ert_fes > ert_fe_max:
                    raise ValueError(
                        "ert_fes must be <= "
                        f"{ert_fe_max}, but is {ert_fes}.")

                ert_time_millis = _try_int(ert_time_millis)
                if ert_time_millis < success_time_millis.minimum:
                    raise ValueError(
                        "ert_time_millis must be >= "
                        f"{success_time_millis.minimum}, but "
                        f"is {ert_time_millis}.")
                ert_time_max = total_time_millis.mean_arith * n
                if ert_time_millis > ert_time_max:
                    raise ValueError(
                        "ert_fes must be <= "
                        f"{ert_time_max}, but is {ert_time_millis}.")
            else:
                if success_fes is not None:
                    raise ValueError(
                        "If n_success<=0, then success_fes must be None, "
                        f"but it's a {type(success_fes)}.")
                if success_time_millis is not None:
                    raise ValueError(
                        "If n_success<=0, then success_time_millis must "
                        f"be None, but it's a {type(success_time_millis)}.")
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
                raise ValueError(
                    f"max_fes must be >= {total_fes.maximum},"
                    f" but is {max_fes}.")
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
            raise TypeError(
                "max_fes must be int, Statistics, or None, but is "
                f"{type(max_fes)}.")
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
            raise TypeError(
                "max_time_millis must be int, Statistics, or None, but is "
                f"{type(max_time_millis)}.")
        object.__setattr__(self, "max_time_millis", max_time_millis)

    @staticmethod
    def create(source: Iterable[EndResult]) -> 'EndStatistics':
        """
        Create an `EndStatistics` Record from an Iterable of `EndResult`.

        :param Iterable[moptipy.evaluation.EndResult] source: the source
        :return: the statistics
        :rtype: EndStatistics
        """
        if not isinstance(source, Iterable):
            raise TypeError(
                f"source must be Iterable, but is {type(source)}.")

        n: int = 0
        best_f: List[Union[int, float]] = list()
        last_improvement_fe: List[int] = list()
        last_improvement_time_millis: List[int] = list()
        total_fes: List[int] = list()
        total_time_millis: List[int] = list()
        max_fes: Optional[List[int]] = list()
        max_fes_same: bool = True
        max_time_millis: Optional[List[int]] = list()
        max_time_same: bool = True
        goal_f: Optional[List[Union[int, float]]] = list()
        goal_f_same: bool = True
        best_f_scaled: Optional[List[float]] = list()
        n_success: Optional[int] = 0
        success_fes: Optional[List[int]] = list()
        success_times: Optional[List[int]] = list()

        fes: int = 0
        time: int = 0
        algorithm: Optional[str] = None
        instance: Optional[str] = None

        for er in source:
            if not isinstance(er, EndResult):
                raise TypeError("Only end results are permitted, but "
                                f"encountered a {type(er)}.")
            if n == 0:
                algorithm = er.algorithm
                instance = er.instance
            else:
                if algorithm != er.algorithm:
                    algorithm = None
                if instance != er.instance:
                    instance = None
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
                    max_fes_same &= (max_fes[-1] == er.max_fes)
                max_fes.append(er.max_fes)
            if er.max_time_millis is None:
                max_time_millis = None
            elif max_time_millis is not None:
                if n > 1:
                    max_time_same &= \
                        (max_time_millis[-1] == er.max_time_millis)
                max_time_millis.append(er.max_time_millis)

            if er.goal_f is None:
                goal_f = None
                best_f_scaled = None
                n_success = None
                success_fes = None
                success_times = None
            else:
                if goal_f is not None:
                    if n > 1:
                        goal_f_same &= (goal_f[-1] == er.goal_f)
                    goal_f.append(er.goal_f)

                    if er.goal_f <= 0:
                        best_f_scaled = None
                    else:
                        if best_f_scaled is not None:
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
            else (inf if (n_success <= 0) else _try_div(fes, n_success)),
            None if (n_success is None) else
            (inf if (n_success <= 0) else _try_div(time, n_success)),
            None if max_fes is None else
            (max_fes[0] if max_fes_same else Statistics.create(max_fes)),
            None if max_time_millis is None
            else (max_time_millis[0] if max_time_same
                  else Statistics.create(max_time_millis)))

    @staticmethod
    def from_end_results(source: Iterable[EndResult],
                         collector: MutableSequence['EndStatistics'],
                         join_all_algorithms: bool = False,
                         join_all_instances: bool = False) -> None:
        """
        Aggregate statistics over a stream of end results.

        :param Iterable[moptipy.evaluation.EndResult] source: the stream
            of end results
        :param MutableSequence['EndStatistic'] collector: the destination
            to which the new records will be appended
        :param bool join_all_algorithms: should the statistics be aggregated
            over all algorithms
        :param bool join_all_instances: should the statistics be aggregated
            over all algorithms
        """
        if not isinstance(source, Iterable):
            raise TypeError(
                f"source must be Iterable, but is {type(source)}.")
        if not isinstance(collector, MutableSequence):
            raise TypeError("collector must be MutableSequence, "
                            f"but is {type(collector)}.")
        if not isinstance(join_all_algorithms, bool):
            raise TypeError("join_all_algorithms must be bool, "
                            f"but is {type(join_all_algorithms)}.")
        if not isinstance(join_all_instances, bool):
            raise TypeError("join_all_instances must be bool, "
                            f"but is {type(join_all_instances)}.")

        if join_all_algorithms and join_all_instances:
            collector.append(EndStatistics.create(source))
            return

        sorter: Dict[str, List[EndResult]] = dict()
        for er in source:
            if not isinstance(er, EndResult):
                raise TypeError("source must contain only EndResults, but "
                                f"found a {type(er)}.")
            key = er.instance if join_all_algorithms else \
                er.algorithm if join_all_instances else \
                f"{er.algorithm}/{er.instance}"
            if key in sorter:
                lst = sorter[key]
            else:
                lst = list()
                sorter[key] = lst
            lst.append(er)

        if len(sorter) <= 0:
            raise ValueError("source must not be empty")

        if len(sorter) > 1:
            keyz = list(sorter.keys())
            keyz.sort()
            for key in keyz:
                collector.append(EndStatistics.create(sorter[key]))
        else:
            collector.append(EndStatistics.create(
                next(iter(sorter.values()))))

    @staticmethod
    def to_csv(data: Union['EndStatistics',
                           Iterable['EndStatistics']], file: str) -> None:
        """
        Store a set of :class:`EndStatistics` in a CSV file.

        :param Union['EndStatistics', Iterable['EndStatistics']] data: the
            data to store
        :param str file: the file to generate
        """
        file = canonicalize_path(file)
        print(f"{datetime.now()}: Writing end result statistics to "
              f"CSV file '{file}'.")

        has_algorithm: bool = False  # 1
        has_instance: bool = False  # 2
        has_goal_f: int = 0  # 4
        has_best_f_scaled: bool = False  # 8
        has_n_success: bool = False  # 16
        has_success_fes: bool = False  # 32
        has_success_time_millis: bool = False  # 64
        has_ert_fes: bool = False  # 128
        has_ert_time_millis: bool = False  # 256
        has_max_fes: int = 0  # 512
        has_max_time_millis: int = 0  # 1024
        checker: int = 2047

        if isinstance(data, EndStatistics):
            data = [data]

        for es in data:
            if es.algorithm is not None:
                has_algorithm = True
                checker &= ~1
            if es.instance is not None:
                has_instance = True
                checker &= ~2
            if es.goal_f is not None:
                if isinstance(es.goal_f, Statistics):
                    has_goal_f = 2
                    checker &= ~4
                elif has_goal_f == 0:
                    has_goal_f = 1
            if es.best_f_scaled is not None:
                has_best_f_scaled = True
                checker &= ~8
            if es.n_success is not None:
                has_n_success = True
                checker &= ~8
            if es.success_fes is not None:
                has_success_fes = True
                checker &= ~32
            if es.success_time_millis is not None:
                has_success_time_millis = True
                checker &= ~64
            if es.ert_fes is not None:
                has_ert_fes = True
                checker &= ~128
            if es.ert_time_millis is not None:
                has_ert_time_millis = True
                checker &= ~256
            if es.max_fes is not None:
                if isinstance(es.max_fes, Statistics):
                    has_max_fes = 2
                    checker &= ~512
                elif has_max_fes == 0:
                    has_max_fes = 1
            if es.max_time_millis is not None:
                if isinstance(es.max_time_millis, Statistics):
                    has_max_time_millis = 2
                    checker &= ~512
                elif has_max_time_millis == 0:
                    has_max_time_millis = 1
            if checker == 0:
                break

        with open(file, "wt") as out:
            wrt: Final[Callable] = out.write
            sep: Final[str] = log.CSV_SEPARATOR
            if has_algorithm:
                wrt(log.KEY_ALGORITHM)
                wrt(sep)
            if has_instance:
                wrt(log.KEY_INSTANCE)
                wrt(sep)

            def h(p):
                wrt(sep.join(Statistics.csv_col_names(p)))

            wrt(KEY_N)
            wrt(sep)
            h(log.KEY_BEST_F)
            wrt(sep)
            h(log.KEY_LAST_IMPROVEMENT_FE)
            wrt(sep)
            h(log.KEY_LAST_IMPROVEMENT_TIME_MILLIS)
            wrt(sep)
            h(log.KEY_TOTAL_FES)
            wrt(sep)
            h(log.KEY_TOTAL_TIME_MILLIS)
            if has_goal_f == 1:
                wrt(sep)
                wrt(log.KEY_GOAL_F)
            elif has_goal_f == 2:
                wrt(sep)
                h(log.KEY_GOAL_F)
            if has_best_f_scaled:
                wrt(sep)
                h(KEY_BEST_F_SCALED)
            if has_n_success:
                wrt(sep)
                wrt(KEY_N_SUCCESS)
            if has_success_fes:
                wrt(sep)
                wrt(KEY_SUCCESS_FES)
            if has_success_time_millis:
                wrt(sep)
                wrt(KEY_SUCCESS_TIME_MILLIS)
            if has_ert_fes:
                wrt(sep)
                wrt(KEY_ERT_FES)
            if has_ert_time_millis:
                wrt(sep)
                wrt(KEY_ERT_TIME_MILLIS)
            if has_max_fes == 1:
                wrt(sep)
                wrt(log.KEY_MAX_FES)
            elif has_max_fes == 2:
                wrt(sep)
                h(log.KEY_MAX_FES)
            if has_max_time_millis == 1:
                wrt(sep)
                wrt(log.KEY_MAX_TIME_MILLIS)
            elif has_max_time_millis == 2:
                wrt(sep)
                h(log.KEY_MAX_TIME_MILLIS)
            out.write("\n")

            csv: Final[Callable] = Statistics.value_to_csv
            num: Final[Callable] = log.num_to_str

            for er in data:
                if has_algorithm:
                    if er.algorithm is not None:
                        wrt(er.algorithm)
                    wrt(sep)
                if has_instance:
                    if er.instance is not None:
                        wrt(er.instance)
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
                        wrt(num(er.goal_f))
                elif has_goal_f == 2:
                    wrt(sep)
                    if isinstance(er.goal_f, Statistics):
                        wrt(er.goal_f.to_csv())
                    elif isinstance(er.goal_f, (int, float)):
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
                        wrt(str(er.max_fes))
                elif has_max_fes == 2:
                    wrt(sep)
                    if isinstance(er.max_fes, Statistics):
                        wrt(er.max_fes.to_csv())
                    elif isinstance(er.max_fes, (int, float)):
                        wrt(csv(er.max_fes))
                    else:
                        wrt(EMPTY_CSV_ROW)
                if has_max_time_millis == 1:
                    wrt(sep)
                    if er.max_time_millis is not None:
                        wrt(str(er.max_time_millis))
                elif has_max_time_millis == 2:
                    wrt(sep)
                    if isinstance(er.max_time_millis, Statistics):
                        wrt(er.max_time_millis.to_csv())
                    elif isinstance(er.max_time_millis, (int, float)):
                        wrt(csv(er.max_time_millis))
                    else:
                        wrt(EMPTY_CSV_ROW)
                out.write("\n")

        print(f"{datetime.now()}: Done writing end result statistics to "
              f"CSV file '{file}'.")

    @staticmethod
    def from_csv(file: str,
                 collector: MutableSequence['EndStatistics']) -> None:
        """
        Parse a CSV file and collect all encountered :class:`EndStatistics`.

        :param str file: the file to parse
        :param MutableSequence['EndStatistics'] collector: the collector to
            receive all the parsed instances of :class:`EndStatistics`.
        """
        file = enforce_file(canonicalize_path(file))
        print(f"{datetime.now()}: Begin reading end result statistics from "
              f"CSV file '{file}'.")

        sep: Final[str] = log.CSV_SEPARATOR
        with open(file, "rt") as rd:
            headerrow: Final[List[str]] = rd.readlines(1)
            if (headerrow is None) or (len(headerrow) <= 0):
                raise ValueError(f"No line in file '{file}'.")
            headerstr: Final[str] = headerrow[0].strip()
            header: Final[List[str]] = [ss.strip()
                                        for ss in headerstr.split(sep)]
            if len(header) <= 3:
                raise ValueError(
                    f"Invalid header '{headerstr}' in file '{file}'.")

            idx = 0
            has_algorithm: bool
            if header[0] == log.KEY_ALGORITHM:
                has_algorithm = True
                idx = 1
            else:
                has_algorithm = False

            has_instance: bool
            if header[idx] == log.KEY_INSTANCE:
                has_instance = True
                idx += 1
            else:
                has_instance = False

            csv: Final[Callable] = Statistics.csv_col_names

            if header[idx] != KEY_N:
                raise ValueError(
                    f"Expected to find {KEY_N} at index {idx} "
                    f"in header '{headerstr}' of file '{file}'.")
            idx += 1

            for key in [log.KEY_BEST_F, log.KEY_LAST_IMPROVEMENT_FE,
                        log.KEY_LAST_IMPROVEMENT_TIME_MILLIS,
                        log.KEY_TOTAL_FES, log.KEY_TOTAL_TIME_MILLIS]:
                if csv(key) != header[idx:(idx + CSV_COLS)]:
                    raise ValueError(
                        f"Expected to find '{key}.*' keys from index "
                        f"{idx} on in header "
                        f"'{headerstr}' of file '{file}', expected "
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
                if header[idx] == log.KEY_GOAL_F:
                    has_goal_f = 1
                    idx += 1
                elif header[idx].startswith(log.KEY_GOAL_F):
                    has_goal_f = 2
                    if csv(log.KEY_GOAL_F) != header[idx:(idx + CSV_COLS)]:
                        raise ValueError(
                            f"Expected to find '{log.KEY_GOAL_F}.*' keys from "
                            f"index {idx} on in header "
                            f"'{headerstr}' of file '{file}'.")
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
                            f"'{headerstr}' of file '{file}'.")
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
                            f"'{headerstr}' of file '{file}'.")
                    idx += CSV_COLS

                if idx >= len(header):
                    break

                if header[idx].startswith(KEY_SUCCESS_TIME_MILLIS):
                    has_success_fes = True
                    if csv(KEY_SUCCESS_TIME_MILLIS) != \
                            header[idx:(idx + CSV_COLS)]:
                        raise ValueError(
                            f"Expected to find '{KEY_SUCCESS_TIME_MILLIS}.*' "
                            f"keys from index {idx} on in header "
                            f"'{headerstr}' of file '{file}'.")
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

                if header[idx] == log.KEY_MAX_FES:
                    has_max_fes = 1
                    idx += 1
                elif header[idx].startswith(log.KEY_MAX_FES):
                    has_max_fes = 2
                    if csv(log.KEY_MAX_FES) != header[idx:(idx + CSV_COLS)]:
                        raise ValueError(
                            f"Expected to find '{log.KEY_MAX_FES}.*' keys"
                            f" from index {idx} on in header "
                            f"'{headerstr}' of file '{file}'.")
                    idx += CSV_COLS

                if idx >= len(header):
                    break

                if header[idx] == log.KEY_MAX_TIME_MILLIS:
                    has_max_time = 1
                    idx += 1
                elif header[idx].startswith(log.KEY_MAX_TIME_MILLIS):
                    has_max_time = 2
                    if csv(log.KEY_MAX_FES) != header[idx:(idx + CSV_COLS)]:
                        raise ValueError(
                            f"Expected to find '{log.KEY_MAX_TIME_MILLIS}.*' "
                            f"keys from index {idx} on in header "
                            f"'{headerstr}' of file '{file}'.")
                    idx += CSV_COLS

                break

            if len(header) > idx:
                raise ValueError(
                    f"Unexpected item '{header[idx]}' in header "
                    f"'{header}' of file '{file}'.")

            while True:
                lines = rd.readlines(100)
                if (lines is None) or (len(lines) <= 0):
                    break
                for line in lines:
                    row = [ss.strip() for ss in line.strip().split(sep)]

                    idx = 0
                    algo: Optional[str] = None
                    inst: Optional[str] = None
                    n: int
                    goal_f: Union[None, int, float, Statistics] = None
                    best_f_scaled: Optional[Statistics] = None
                    n_success: Optional[int] = None
                    success_fes: Optional[Statistics] = None
                    success_time: Optional[Statistics] = None
                    ert_fes: Union[int, float, None] = None
                    ert_time: Union[int, float, None] = None
                    max_fes: Union[int, Statistics, None] = None
                    max_time: Union[int, Statistics, None] = None

                    try:
                        if has_algorithm:
                            algo = log.sanitize_name(row[0])
                            idx += 1

                        if has_instance:
                            inst = log.sanitize_name(row[idx])
                            idx += 1

                        n = int(row[idx])
                        idx += 1

                        best_f = Statistics.from_csv(
                            n, row[idx:(idx + CSV_COLS)])
                        idx += CSV_COLS

                        last_improv_fe = Statistics.from_csv(
                            n, row[idx:(idx + CSV_COLS)])
                        idx += CSV_COLS

                        last_improv_time = Statistics.from_csv(
                            n, row[idx:(idx + CSV_COLS)])
                        idx += CSV_COLS

                        total_fes = Statistics.from_csv(
                            n, row[idx:(idx + CSV_COLS)])
                        idx += CSV_COLS

                        total_time = Statistics.from_csv(
                            n, row[idx:(idx + CSV_COLS)])
                        idx += CSV_COLS

                        if has_goal_f == 1:
                            goal_f = _str_to_if(row[idx])
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
                                n_success, row[idx:(idx + CSV_COLS)])
                            idx += CSV_COLS

                        if has_success_time:
                            success_time = Statistics.from_csv(
                                n_success, row[idx:(idx + CSV_COLS)])
                            idx += CSV_COLS

                        if has_ert_fes:
                            ert_fes = _str_to_if(row[idx])
                            idx += 1

                        if has_ert_time:
                            ert_time = _str_to_if(row[idx])
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

                    except BaseException as be:
                        raise ValueError(
                            f"Invalid row '{line}' in file '{file}'.") from be

                    if len(row) != idx:
                        raise ValueError("Invalid number of columns in row "
                                         f"'{line}' in file '{file}'.")
                    collector.append(EndStatistics(
                        algo, inst, n, best_f,
                        last_improv_fe, last_improv_time, total_fes,
                        total_time, goal_f, best_f_scaled, n_success,
                        success_fes, success_time, ert_fes, ert_time,
                        max_fes, max_time))

        print(f"{datetime.now()}: Finished reading end result statistics "
              f"from CSV file '{file}'.")
