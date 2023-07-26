"""
Record for EndResult as well as parsing, serialization, and parsing.

When doing experiments with `moptipy`, you apply algorithm setups to problem
instances. For each `setup x instance` combination, you may conduct a series
of repetitions (so-called runs) with different random seeds. Each single run
of an algorithm setup on a problem instances can produce a separate log file.
From each log file, we can load a :class:`EndResult` instance, which
represents, well, the end result of the run, i.e., information such as the
best solution quality reached, when it was reached, and the termination
criterion. These end result records then can be the basis for, e.g., computing
summary statistics via :mod:`~moptipy.evaluation.end_statistics` or for
plotting the end result distribution via
:mod:`~moptipy.evaluation.plot_end_results`.
"""
import argparse
import os.path
from dataclasses import dataclass
from math import inf, isfinite
from typing import Any, Callable, Final, Iterable, cast

from moptipy.api.logging import (
    FILE_SUFFIX,
    KEY_ALGORITHM,
    KEY_BEST_F,
    KEY_GOAL_F,
    KEY_INSTANCE,
    KEY_LAST_IMPROVEMENT_FE,
    KEY_LAST_IMPROVEMENT_TIME_MILLIS,
    KEY_MAX_FES,
    KEY_MAX_TIME_MILLIS,
    KEY_RAND_SEED,
    KEY_TOTAL_FES,
    KEY_TOTAL_TIME_MILLIS,
    PROGRESS_CURRENT_F,
    PROGRESS_FES,
    PROGRESS_TIME_MILLIS,
    SECTION_PROGRESS,
)
from moptipy.evaluation._utils import (
    _check_max_time_millis,
)
from moptipy.evaluation.base import (
    F_NAME_NORMALIZED,
    F_NAME_RAW,
    F_NAME_SCALED,
    PerRunData,
)
from moptipy.evaluation.log_parser import SetupAndStateParser
from moptipy.utils.console import logger
from moptipy.utils.help import argparser
from moptipy.utils.logger import CSV_SEPARATOR
from moptipy.utils.math import try_float_div, try_int
from moptipy.utils.path import Path
from moptipy.utils.strings import (
    intfloatnone_to_str,
    intnone_to_str,
    num_to_str,
    sanitize_names,
    str_to_intfloat,
    str_to_intfloatnone,
    str_to_intnone,
)
from moptipy.utils.types import check_int_range, check_to_int_range, type_error

#: The internal CSV header
_HEADER: Final[str] = (f"{KEY_ALGORITHM}{CSV_SEPARATOR}"
                       f"{KEY_INSTANCE}{CSV_SEPARATOR}"
                       f"{KEY_RAND_SEED}{CSV_SEPARATOR}"
                       f"{KEY_BEST_F}{CSV_SEPARATOR}"
                       f"{KEY_LAST_IMPROVEMENT_FE}{CSV_SEPARATOR}"
                       f"{KEY_LAST_IMPROVEMENT_TIME_MILLIS}"
                       f"{CSV_SEPARATOR}"
                       f"{KEY_TOTAL_FES}{CSV_SEPARATOR}"
                       f"{KEY_TOTAL_TIME_MILLIS}{CSV_SEPARATOR}"
                       f"{KEY_GOAL_F}{CSV_SEPARATOR}"
                       f"{KEY_MAX_FES}{CSV_SEPARATOR}"
                       f"{KEY_MAX_TIME_MILLIS}\n")


def __get_goal_f(e: "EndResult") -> int | float:
    """
    Get the goal_f.

    :param e: the end result
    :returns: the goal objective value
    """
    g = e.goal_f
    if g is None:
        raise ValueError(f"goal_f of {e} is None!")
    if not isfinite(g):
        raise ValueError(f"goal_f {g} of {e} is not finite!")
    return g


def __get_max_fes(e: "EndResult") -> int | float:
    """
    Get the max FEs.

    :param e: the end result
    :returns: the max fes
    """
    g = e.max_fes
    if g is None:
        raise ValueError(f"max_fes of {e} is None!")
    return g


def __get_max_time_millis(e: "EndResult") -> int | float:
    """
    Get the maximum time in milliseconds.

    :param e: the end result
    :returns: the maximum time in milliseconds
    """
    g = e.max_time_millis
    if g is None:
        raise ValueError(f"max_time_millis of {e} is None!")
    return g


def __get_goal_f_for_div(e: "EndResult") -> int | float:
    """
    Get the goal_f.

    :param e: the end result
    :returns: the goal objective value
    """
    g = __get_goal_f(e)
    if g <= 0:
        raise ValueError(f"goal_f {g} of {e}is not positive!")
    return g


def __get_f_norm(e: "EndResult") -> int | float:
    """
    Get the normalized f.

    :param e: the end result
    :returns: the normalized f
    """
    g = __get_goal_f_for_div(e)
    return try_float_div(e.best_f - g, g)


#: A set of getters for accessing variables of the end result
_GETTERS: Final[dict[str, Callable[["EndResult"], int | float]]] = {
    KEY_LAST_IMPROVEMENT_FE: lambda e: e.last_improvement_fe,
    KEY_LAST_IMPROVEMENT_TIME_MILLIS:
        lambda e: e.last_improvement_time_millis,
    KEY_TOTAL_FES: lambda e: e.total_fes,
    KEY_TOTAL_TIME_MILLIS: lambda e: e.total_time_millis,
    KEY_GOAL_F: __get_goal_f,
    F_NAME_RAW: lambda e: e.best_f,
    F_NAME_SCALED: lambda e: try_float_div(e.best_f, __get_goal_f_for_div(e)),
    F_NAME_NORMALIZED: __get_f_norm,
    KEY_MAX_FES: __get_max_fes,
    KEY_MAX_TIME_MILLIS: __get_max_time_millis,
}
_GETTERS[KEY_BEST_F] = _GETTERS[F_NAME_RAW]


@dataclass(frozen=True, init=False, order=True)
class EndResult(PerRunData):
    """
    An immutable end result record of one run of one algorithm on one problem.

    This record provides the information of the outcome of one application of
    one algorithm to one problem instance in an immutable way.
    """

    #: The best objective value encountered.
    best_f: int | float

    #: The index of the function evaluation when best_f was reached.
    last_improvement_fe: int

    #: The time when best_f was reached.
    last_improvement_time_millis: int

    #: The total number of performed FEs.
    total_fes: int

    #: The total time consumed by the run.
    total_time_millis: int

    #: The goal objective value if provided
    goal_f: int | float | None

    #: The (optional) maximum permitted FEs.
    max_fes: int | None

    #: The (optional) maximum runtime.
    max_time_millis: int | None

    def __init__(self,
                 algorithm: str,
                 instance: str,
                 rand_seed: int,
                 best_f: int | float,
                 last_improvement_fe: int,
                 last_improvement_time_millis: int,
                 total_fes: int,
                 total_time_millis: int,
                 goal_f: int | float | None,
                 max_fes: int | None,
                 max_time_millis: int | None):
        """
        Create a consistent instance of :class:`EndResult`.

        :param algorithm: the algorithm name
        :param instance: the instance name
        :param rand_seed: the random seed
        :param best_f: the best reached objective value
        :param last_improvement_fe: the FE when best_f was reached
        :param last_improvement_time_millis: the time when best_f was reached
        :param total_fes: the total FEs
        :param total_time_millis: the total runtime
        :param goal_f: the goal objective value, if provide
        :param max_fes: the optional maximum FEs
        :param max_time_millis: the optional maximum runtime

        :raises TypeError: if any parameter has a wrong type
        :raises ValueError: if the parameter values are inconsistent
        """
        super().__init__(algorithm, instance, rand_seed)
        object.__setattr__(self, "best_f", try_int(best_f))
        object.__setattr__(
            self, "last_improvement_fe", check_int_range(
                last_improvement_fe, "last_improvement_fe",
                1, 1_000_000_000_000_000))
        object.__setattr__(
            self, "last_improvement_time_millis", check_int_range(
                last_improvement_time_millis, "last_improvement_time_millis",
                0, 100_000_000_000))
        object.__setattr__(
            self, "total_fes", check_int_range(
                total_fes, "total_fes", last_improvement_fe,
                1_000_000_000_000_000))
        object.__setattr__(
            self, "total_time_millis", check_int_range(
                total_time_millis, "total_time_millis",
                last_improvement_time_millis, 100_000_000_000))

        if goal_f is not None:
            goal_f = None if goal_f <= -inf else try_int(goal_f)
        object.__setattr__(self, "goal_f", goal_f)

        if max_fes is not None:
            check_int_range(max_fes, "max_fes", total_fes,
                            1_000_000_000_000_000_000)
        object.__setattr__(self, "max_fes", max_fes)

        if max_time_millis is not None:
            check_int_range(
                max_time_millis, "max_time_millis", 1, 100_000_000_000)
            _check_max_time_millis(max_time_millis,
                                   total_fes,
                                   total_time_millis)
        object.__setattr__(self, "max_time_millis", max_time_millis)

    def success(self) -> bool:
        """
        Check if a run is successful.

        This method returns `True` if and only if `goal_f` is defined and
        `best_f <= goal_f` (and `False` otherwise).

        :return: `True` if and only if `best_f<=goal_f`
        """
        return False if self.goal_f is None else self.best_f <= self.goal_f

    def path_to_file(self, base_dir: str) -> Path:
        """
        Get the path that would correspond to the log file of this end result.

        Obtain a path that would correspond to the log file of this end
        result, resolved from a base directory `base_dir`.

        :param base_dir: the base directory
        :returns: the path to a file corresponding to the end result record
        """
        return Path.path(base_dir).resolve_inside(
            self.algorithm).resolve_inside(self.instance).resolve_inside(
            sanitize_names([self.algorithm, self.instance,
                            hex(self.rand_seed)]) + FILE_SUFFIX)

    @staticmethod
    def getter(dimension: str) -> Callable[["EndResult"], int | float]:
        """
        Produce a function that obtains the given dimension from EndResults.

        The following dimensions are supported:

        1. `lastImprovementFE`: :attr:`~EndResult.last_improvement_fe`
        2. `lastImprovementTimeMillis`:
            :attr:`~EndResult.last_improvement_time_millis`
        3. `totalFEs`: :attr:`~EndResult.total_fes`
        4. `totalTimeMillis`: :attr:`~EndResult.total_time_millis`
        5. `goalF`: :attr:`~EndResult.goal_f`
        6.  `plainF`, `bestF`: :attr:`~EndResult.best_f`
        7. `scaledF`: :attr:`~EndResult.best_f`/:attr:`~EndResult.goal_f`
        8. `normalizedF`: (:attr:`~EndResult.best_f`-attr:`~EndResult.goal_f`)/
            :attr:`~EndResult.goal_f`
        9. `maxFEs`: :attr:`~EndResult.max_fes`
        10. `maxTimeMillis`: :attr:`~EndResult.max_time_millis`
        11. `fesPerTimeMilli`:  :attr:`~EndResult.total_fes`
            /:attr:`~EndResult.total_time_millis`

        :param dimension: the dimension
        :returns: a callable that returns the value corresponding to the
            dimension from its input value, which must be an :class:`EndResult`
        """
        if not isinstance(dimension, str):
            raise type_error(dimension, "dimension", str)
        if dimension in _GETTERS:
            return _GETTERS[dimension]
        raise ValueError(f"unknown dimension {dimension!r}, "
                         f"should be one of {sorted(_GETTERS.keys())}.")

    @staticmethod
    def from_logs(
            path: str, consumer: Callable[["EndResult"], Any],
            max_fes: int | None | Callable[
                [str, str], int | None] = None,
            max_time_millis: int | None | Callable[
                [str, str], int | None] = None,
            goal_f: int | float | None | Callable[
                [str, str], int | float | None] = None) -> None:
        """
        Parse a given path and pass all end results found to the consumer.

        If `path` identifies a file with suffix `.txt`, then this file is
        parsed. The appropriate :class:`EndResult` is created and appended to
        the `collector`. If `path` identifies a directory, then this directory
        is parsed recursively for each log file found, one record is passed to
        the `consumer`. As `consumer`, you could pass any `callable` that
        accepts instances of :class:`EndResult`, e.g., the `append` method of
        a :class:`list`.

        Via the parameters `max_fes`, `max_time_millis`, and `goal_f`, you can
        set virtual limits for the objective function evaluations, the maximum
        runtime, and the objective value. The :class:`EndResult` records will
        then not represent the actual final state of the runs but be
        synthesized from the logged progress information. This, of course,
        requires such information to be present. It will also raise a
        `ValueError` if the goals are invalid, e.g., if a runtime limit is
        specified that is before the first logged points.

        There is one caveat when specifying `max_time_millis`: Let's say that
        the log files only log improvements. Then you might have a log point
        for 7000 FEs, 1000ms, and f=100. The next log point could be 8000 FEs,
        1200ms, and f=90. Now if your time limit specified is 1100ms, we know
        that the end result is f=100 (because f=90 was reached too late) and
        that the total runtime is 1100ms, as this is the limit you specified
        and it was also reached. But we do not know the number of consumed
        FEs. We know you consumed at least 7000 FEs, but you did not consume
        8000 FEs. It would be wrong to claim that 7000 FEs were consumed,
        since it could have been more. We therefore set a virtual end point at
        7999 FEs. In terms of performance metrics such as the
        :mod:`~moptipy.evaluation.ert`, this would be the most conservative
        choice in that it does not over-estimate the speed of the algorithm.
        It can, however, lead to very big deviations from the actual values.
        For example, if your algorithm quickly converged to a local optimum
        and there simply is no log point that exceeds the virtual time limit
        but the original run had a huge FE-based budget while your virtual
        time limit was small, this could lead to an estimate of millions of
        FEs taking part within seconds...

        :param path: the path to parse
        :param consumer: the consumer
        :param max_fes: the maximum FEs, a callable to compute the maximum
            FEs from the algorithm and instance name, or `None` if unspecified
        :param max_time_millis: the maximum runtime in milliseconds, a
            callable to compute the maximum runtime from the algorithm and
            instance name, or `None` if unspecified
        :param goal_f: the goal objective value, a callable to compute the
            goal objective value from the algorithm and instance name, or
            `None` if unspecified
        """
        need_goals: bool = False
        if max_fes is not None:
            if not callable(max_fes):
                max_fes = check_int_range(
                    max_fes, "max_fes", 1, 1_000_000_000_000_000)
            need_goals = True
        if max_time_millis is not None:
            if not callable(max_time_millis):
                max_time_millis = check_int_range(
                    max_time_millis, "max_time_millis", 1, 1_000_000_000_000)
            need_goals = True
        if goal_f is not None:
            if callable(goal_f):
                need_goals = True
            else:
                if not isinstance(goal_f, int | float):
                    raise type_error(goal_f, "goal_f", (int, float, None))
                if isfinite(goal_f):
                    need_goals = True
                elif goal_f <= -inf:
                    goal_f = None
                else:
                    raise ValueError(f"goal_f={goal_f} is not permissible.")
        if need_goals:
            _InnerProgressLogParser(
                max_fes, max_time_millis, goal_f, consumer).parse(path)
        else:
            _InnerLogParser(consumer).parse(path)

    @staticmethod
    def to_csv(results: Iterable["EndResult"], file: str) -> Path:
        """
        Write a sequence of end results to a file in CSV format.

        :param results: the end results
        :param file: the path
        :return: the path of the file that was written
        """
        path: Final[Path] = Path.path(file)
        logger(f"Writing end results to CSV file {path!r}.")
        Path.path(os.path.dirname(path)).ensure_dir_exists()

        with path.open_for_write() as out:
            out.write(_HEADER)
            for e in results:
                out.write(
                    f"{e.algorithm}{CSV_SEPARATOR}"
                    f"{e.instance}{CSV_SEPARATOR}"
                    f"{hex(e.rand_seed)}{CSV_SEPARATOR}"
                    f"{num_to_str(e.best_f)}{CSV_SEPARATOR}"
                    f"{e.last_improvement_fe}{CSV_SEPARATOR}"
                    f"{e.last_improvement_time_millis}{CSV_SEPARATOR}"
                    f"{e.total_fes}{CSV_SEPARATOR}"
                    f"{e.total_time_millis}{CSV_SEPARATOR}"
                    f"{intfloatnone_to_str(e.goal_f)}{CSV_SEPARATOR}"
                    f"{intnone_to_str(e.max_fes)}{CSV_SEPARATOR}"
                    f"{intnone_to_str(e.max_time_millis)}\n")

        logger(f"Done writing end results to CSV file {path!r}.")
        return path

    @staticmethod
    def from_csv(file: str, consumer: Callable[["EndResult"], Any],
                 filterer: Callable[["EndResult"], bool]
                 = lambda x: True) -> None:
        """
        Parse a given CSV file to get :class:`EndResult` Records.

        :param file: the path to parse
        :param consumer: the collector, can be the `append` method of a
            :class:`list`
        :param filterer: an optional filter function
        """
        if not callable(consumer):
            raise type_error(consumer, "consumer", call=True)
        path: Final[Path] = Path.file(file)
        logger(f"Now reading CSV file {path!r}.")

        with path.open_for_read() as rd:
            header = rd.readlines(1)
            if (header is None) or (len(header) <= 0):
                raise ValueError(f"No line in file {file!r}.")
            if header[0] != _HEADER:
                raise ValueError(
                    f"Header {header[0]!r} in {path!r} should be {_HEADER!r}.")

            while True:
                lines = rd.readlines(100)
                if (lines is None) or (len(lines) <= 0):
                    break
                for line in lines:
                    splt = line.strip().split(CSV_SEPARATOR)
                    er = EndResult(
                        splt[0].strip(),  # algorithm
                        splt[1].strip(),  # instance
                        int((splt[2])[2:], 16),  # rand seed
                        str_to_intfloat(splt[3]),  # best_f
                        int(splt[4]),  # last_improvement_fe
                        int(splt[5]),  # last_improvement_time_millis
                        int(splt[6]),  # total_fes
                        int(splt[7]),  # total_time_millis
                        str_to_intfloatnone(splt[8]),  # goal_f
                        str_to_intnone(splt[9]),  # max_fes
                        str_to_intnone(splt[10]))  # max_time_millis
                    if filterer(er):
                        consumer(er)

        logger(f"Done reading CSV file {path!r}.")


class _InnerLogParser(SetupAndStateParser):
    """The internal log parser class."""

    def __init__(self, consumer: Callable[[EndResult], Any]):
        """
        Create the internal log parser.

        :param consumer: the consumer accepting the parsed data
        """
        super().__init__()
        if not callable(consumer):
            raise type_error(consumer, "consumer", call=True)
        self.__consumer: Final[Callable[[EndResult], Any]] = consumer

    def process(self) -> None:
        self.__consumer(EndResult(self.algorithm,
                                  self.instance,
                                  self.rand_seed,
                                  self.best_f,
                                  self.last_improvement_fe,
                                  self.last_improvement_time_millis,
                                  self.total_fes,
                                  self.total_time_millis,
                                  self.goal_f,
                                  self.max_fes,
                                  self.max_time_millis))


def _join_goals(vlimit, vgoal, select):  # noqa
    if vlimit is None:
        return vgoal
    if vgoal is None:
        return vlimit
    return select(vlimit, vgoal)


class _InnerProgressLogParser(SetupAndStateParser):
    """The internal log parser class for virtual end results."""

    def __init__(
            self,
            max_fes: int | None | Callable[[str, str], int | None],
            max_time_millis: int | None | Callable[[str, str], int | None],
            goal_f: int | float | None | Callable[
                [str, str], int | float | None],
            consumer: Callable[[EndResult], Any]):
        """
        Create the internal log parser.

        :param consumer: the consumer
        :param max_fes: the maximum FEs, or `None` if unspecified
        :param max_time_millis: the maximum runtime in milliseconds, or
            `None` if unspecified
        :param goal_f: the goal objective value, or `None` if unspecified
        """
        super().__init__()
        if not callable(consumer):
            raise type_error(consumer, "consumer", call=True)
        self.__consumer: Final[Callable[[EndResult], Any]] = consumer

        self.__src_limit_ms: Final[
            int | None | Callable[[str, str], int | None]] = max_time_millis
        self.__src_limit_fes: Final[
            int | None | Callable[[str, str], int | None]] = max_fes
        self.__src_limit_f: Final[
            int | float | None | Callable[
                [str, str], int | float | None]] = goal_f

        self.__limit_ms: int | float = inf
        self.__limit_ms_n: int | None = None
        self.__limit_fes: int | float = inf
        self.__limit_fes_n: int | None = None
        self.__limit_f: int | float = -inf
        self.__limit_f_n: int | float | None = None

        self.__stop_fes: int | None = None
        self.__stop_ms: int | None = None
        self.__stop_f: int | float | None = None
        self.__stop_li_fe: int | None = None
        self.__stop_li_ms: int | None = None
        self.__hit_goal: bool = False
        self.__state: int = 0

    def end_file(self) -> bool:
        if self.__state != 2:
            raise ValueError(
                "Illegal state, log file must have a "
                f"{SECTION_PROGRESS!r} section.")
        self.__state = 0
        return super().end_file()

    def process(self) -> None:
        hit_goal = self.__hit_goal
        stop_fes: int = self.__stop_fes
        stop_ms: int = self.__stop_ms
        if not hit_goal:
            stop_ms = max(stop_ms, cast(int, min(
                self.total_time_millis, self.__limit_ms)))
            ul_fes = self.total_fes
            if stop_ms < self.total_time_millis:
                ul_fes = ul_fes - 1
            stop_fes = max(stop_fes, cast(int, min(
                ul_fes, self.__limit_fes)))

        self.__consumer(EndResult(
            algorithm=self.algorithm,
            instance=self.instance,
            rand_seed=self.rand_seed,
            best_f=self.__stop_f,
            last_improvement_fe=self.__stop_li_fe,
            last_improvement_time_millis=self.__stop_li_ms,
            total_fes=stop_fes,
            total_time_millis=stop_ms,
            goal_f=_join_goals(self.__limit_f_n, self.goal_f, max),
            max_fes=_join_goals(self.__limit_fes_n, self.max_fes, min),
            max_time_millis=_join_goals(
                self.__limit_ms_n, self.max_time_millis, min)))
        self.__stop_fes = None
        self.__stop_ms = None
        self.__stop_f = None
        self.__stop_li_fe = None
        self.__stop_li_ms = None
        self.__limit_fes_n = None
        self.__limit_fes = inf
        self.__limit_ms_n = None
        self.__limit_ms = inf
        self.__limit_f_n = None
        self.__limit_f = -inf
        self.__hit_goal = False

    def start_file(self, path: Path) -> bool:
        if super().start_file(path):
            if (self.algorithm is None) or (self.instance is None):
                raise ValueError(
                    f"Invalid state: algorithm={self.algorithm!r}, "
                    f"instance={self.instance!r}.")

            fes = self.__src_limit_fes(self.algorithm, self.instance) \
                if callable(self.__src_limit_fes) else self.__src_limit_fes
            self.__limit_fes_n = None if fes is None else \
                check_int_range(fes, "limit_fes", 1, 1_000_000_000_000_000)
            self.__limit_fes = inf if self.__limit_fes_n is None \
                else self.__limit_fes_n

            time = self.__src_limit_ms(self.algorithm, self.instance) \
                if callable(self.__src_limit_ms) else self.__src_limit_ms
            self.__limit_ms_n = None if time is None else \
                check_int_range(time, "limit_ms", 1, 1_000_000_000_000)
            self.__limit_ms = inf if self.__limit_ms_n is None \
                else self.__limit_ms_n

            self.__limit_f_n = self.__src_limit_f(
                self.algorithm, self.instance) \
                if callable(self.__src_limit_f) else self.__src_limit_f
            if self.__limit_f_n is not None:
                if not isinstance(self.__limit_f_n, int | float):
                    raise type_error(self.__limit_f_n, "limit_f", (
                        int, float))
                if not isfinite(self.__limit_f_n):
                    if self.__limit_f_n <= -inf:
                        self.__limit_f_n = None
                    else:
                        raise ValueError(
                            f"invalid limit f={self.__limit_f_n} for "
                            f"{self.algorithm} on {self.instance}")
            self.__limit_f = -inf if self.__limit_f_n is None \
                else self.__limit_f_n
            return True
        return False

    def start_section(self, title: str) -> bool:
        if title == SECTION_PROGRESS:
            if self.__state != 0:
                raise ValueError(f"Already did section {title}.")
            self.__state = 1
            return True
        return super().start_section(title)

    def needs_more_lines(self) -> bool:
        return (self.__state < 2) or super().needs_more_lines()

    def lines(self, lines: list[str]) -> bool:
        if not isinstance(lines, list):
            raise type_error(lines, "lines", list)

        if self.__state != 1:
            return super().lines(lines)
        self.__state = 2

        n_rows = len(lines)
        if n_rows < 2:
            raise ValueError("lines must contain at least two elements,"
                             f"but contains {n_rows}.")

        columns = [c.strip() for c in lines[0].split(CSV_SEPARATOR)]
        fe_col: Final[int] = columns.index(PROGRESS_FES)
        ms_col: Final[int] = columns.index(PROGRESS_TIME_MILLIS)
        f_col: Final[int] = columns.index(PROGRESS_CURRENT_F)
        current_fes: int = -1
        current_ms: int = -1
        current_f: int | float = inf
        current_li_fe: int | None = None
        current_li_ms: int | None = None
        stop_fes: int | None = None
        stop_ms: int | None = None
        stop_f: int | float | None = None
        stop_li_fe: int | None = None
        stop_li_ms: int | None = None
        limit_fes: Final[int | float] = self.__limit_fes
        limit_ms: Final[int | float] = self.__limit_ms
        limit_f: Final[int | float] = self.__limit_f

        for line in lines[1:]:
            values = line.split(CSV_SEPARATOR)
            current_fes = check_to_int_range(
                values[fe_col], "fes", current_fes, 1_000_000_000_000_000)
            current_ms = check_to_int_range(
                values[ms_col], "ms", current_ms, 1_000_000_000_00)
            f: int | float = str_to_intfloat(values[f_col])
            if (current_fes <= limit_fes) and (current_ms <= limit_ms):
                if f < current_f:  # can only update best within budget
                    current_f = f
                    current_li_fe = current_fes
                    current_li_ms = current_ms
                stop_ms = current_ms
                stop_fes = current_fes
                stop_f = current_f
                stop_li_fe = current_li_fe
                stop_li_ms = current_li_ms
            if (current_fes >= limit_fes) or (current_ms >= limit_ms) or \
                    (current_f <= limit_f):
                self.__hit_goal = True
                break  # we can stop parsing the stuff

        if (stop_fes is None) or (stop_ms is None) or (stop_f is None) \
                or (current_fes <= 0) or (not isfinite(current_f)):
            raise ValueError(
                "Illegal state, no fitting data point found: stop_fes="
                f"{stop_fes}, stop_ms={stop_ms}, stop_f={stop_f}, "
                f"current_fes={current_fes}, current_ms={current_ms}, "
                f"current_f={current_f}.")

        if current_fes >= limit_fes:
            stop_fes = max(stop_fes, min(
                cast(int, limit_fes), current_fes))
        elif current_ms > limit_ms:
            stop_fes = max(stop_fes, current_fes - 1)
        else:
            stop_fes = max(stop_fes, current_fes)

        if current_ms >= limit_ms:
            stop_ms = max(stop_ms, min(cast(int, limit_ms), current_ms))
        else:
            stop_ms = max(stop_ms, current_ms)

        self.__stop_fes = stop_fes
        self.__stop_ms = stop_ms
        self.__stop_f = stop_f
        self.__stop_li_fe = stop_li_fe
        self.__stop_li_ms = stop_li_ms
        return self.needs_more_lines()


# Run log files to end results if executed as script
if __name__ == "__main__":
    parser: Final[argparse.ArgumentParser] = argparser(
        __file__,
        "Convert log files obtained with moptipy to the end results CSV "
        "format that can be post-processed or exported to other tools.",
        "This program recursively parses a folder hierarchy created by"
        " the moptipy experiment execution facility. This folder "
        "structure follows the scheme of algorithm/instance/log_file "
        "and has one log file per run. As result of the parsing, one "
        "CSV file (where columns are separated by ';') is created with"
        " one row per log file. This row contains the end-of-run state"
        " loaded from the log file. Whereas the log files may store "
        "the complete progress of one run of one algorithm on one "
        "problem instance as well as the algorithm configuration "
        "parameters, instance features, system settings, and the final"
        " results, the end results CSV file will only represent the "
        "final result quality, when it was obtained, how long the runs"
        " took, etc. This information is much denser and smaller and "
        "suitable for importing into other tools such as Excel or for "
        "postprocessing.")
    parser.add_argument(
        "source", nargs="?", default="./results",
        help="the location of the experimental results, i.e., the root folder "
             "under which to search for log files", type=Path.path)
    parser.add_argument(
        "dest", help="the path to the end results CSV file to be created",
        type=Path.path, nargs="?", default="./evaluation/end_results.txt")
    parser.add_argument(
        "--maxFEs", help="the maximum permitted FEs",
        type=int, nargs="?", default=None)
    parser.add_argument(
        "--maxTime", help="the maximum permitted time in milliseconds",
        type=int, nargs="?", default=None)
    parser.add_argument(
        "--goalF", help="the goal objective value",
        type=str_to_intfloat, nargs="?", default=None)
    args: Final[argparse.Namespace] = parser.parse_args()

    end_results: Final[list[EndResult]] = []
    EndResult.from_logs(args.source, end_results.append,
                        args.maxFEs, args.maxTime, args.goalF)
    EndResult.to_csv(end_results, args.dest)
