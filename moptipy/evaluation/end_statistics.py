"""Statistics aggregated over multiple instances of :class:`EndResult`."""
from dataclasses import dataclass
from math import inf
from typing import Optional, Union, Iterable, List, MutableSequence, Dict

from moptipy.evaluation._utils import _try_int, _try_div
from moptipy.evaluation.end_results import EndResult
from moptipy.evaluation.statistics import Statistics
from moptipy.utils.logging import sanitize_name


@dataclass(frozen=True, init=False, order=True)
class EndStatistics:
    """
    Statistics over end results of one or multiple algorithm*instance setups.

    If one algorithm*instance is used, then `algorithm` and `instance` are
    defined. Otherwise, only the parameter which is the same over all recorded
    runs is defined.
    """

    #: The algorithm that was applied, if the same over all runs.
    algorithm: Optional[str]
    #: The problem instance that was solved, if the same over all runs.
    instance: Optional[str]
    #: The number of runs.
    n: int
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
        if algorithm is not None:
            if algorithm != sanitize_name(algorithm):
                raise ValueError(f"Invalid algorithm '{algorithm}'.")
        object.__setattr__(self, "algorithm", algorithm)

        if instance is not None:
            if instance != sanitize_name(instance):
                raise ValueError(f"Invalid instance '{instance}'.")
        object.__setattr__(self, "instance", instance)

        if not isinstance(n, int):
            raise TypeError(f"n must be int, but is {type(n)}.")
        if n <= 0:
            raise ValueError(f"n must be > 0, but is {n}.")
        object.__setattr__(self, "n", n)

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
            if max_time_millis < total_time_millis.maximum:
                raise ValueError(
                    f"max_time_millis must be >= {total_time_millis.maximum},"
                    f" but is {max_time_millis}.")
        elif isinstance(max_time_millis, Statistics):
            if max_time_millis.minimum < total_time_millis.minimum:
                raise ValueError(
                    "max_time_millis.minimum must be >="
                    f" {total_time_millis.minimum},"
                    f" but is {max_time_millis.minimum}.")
            if max_time_millis.maximum < total_time_millis.maximum:
                raise ValueError(
                    "max_time_millis.maximum must be "
                    f">= {total_time_millis.maximum},"
                    f" but is {max_time_millis.maximum}.")
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

        :param Iterable[.moptipy.evaluation.EndResult] source: the stream
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
        if len(sorter) <= 1:
            collector.append(EndStatistics.create(source))
            return

        keys = list(sorter.keys())
        keys.sort()

        for key in keys:
            collector.append(EndStatistics.create(sorter[key]))
