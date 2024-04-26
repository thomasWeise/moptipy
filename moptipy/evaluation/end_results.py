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
from dataclasses import dataclass
from math import inf, isfinite
from typing import Any, Callable, Final, Iterable, cast

from pycommons.io.console import logger
from pycommons.io.csv import (
    CSV_SEPARATOR,
    SCOPE_SEPARATOR,
    csv_column,
    csv_column_or_none,
    csv_read,
    csv_scope,
    csv_str_or_none,
    csv_val_or_none,
    csv_write,
)
from pycommons.io.path import Path, file_path, line_writer
from pycommons.strings.string_conv import (
    int_or_none_to_str,
    num_or_none_to_str,
    num_to_str,
    str_to_num,
)
from pycommons.types import (
    check_int_range,
    check_to_int_range,
    type_error,
)

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
    DESC_ALGORITHM,
    DESC_ENCODING,
    DESC_INSTANCE,
    DESC_OBJECTIVE_FUNCTION,
    F_NAME_NORMALIZED,
    F_NAME_RAW,
    F_NAME_SCALED,
    KEY_ENCODING,
    KEY_OBJECTIVE_FUNCTION,
    PerRunData,
    _csv_motipy_footer,
)
from moptipy.evaluation.log_parser import SetupAndStateParser
from moptipy.utils.help import moptipy_argparser
from moptipy.utils.math import try_float_div, try_int, try_int_div
from moptipy.utils.strings import (
    sanitize_names,
)

#: a description of the random seed
DESC_RAND_SEED: Final[str] = (
    "the value of the seed of the random number generator used in the run. "
    f"Random seeds are in 0..{int((1 << (8 * 8)) - 1)} and the random "
    f"number generators are those from numpy.")
#: the description of best-F
DESC_BEST_F: Final[str] = (
    " the best (smallest) objective value ever encountered during the run ("
    "regardless whether the algorithm later forgot it again or not).")
#: the description of the last improvement FE
DESC_LAST_IMPROVEMENT_FE: Final[str] = (
    "the objective function evaluation (FE) when the last improving move took"
    " place. 1 FE corresponds to the construction and evaluation "
    "of one solution. The first FE has index 1. With 'last "
    "improving move' we mean the last time when a solution was "
    "discovered that was better than all previous solutions. This "
    "time / FE index is the one when the solution with objective "
    f"value {KEY_BEST_F} was discovered.")
#: the description of the last improvement time milliseconds
DESC_LAST_IMPROVEMENT_TIME_MILLIS: Final[str] = (
    "the clock time in milliseconds after the begin of the run when "
    "the last improving search move took place.")
#: the description of the total FEs
DESC_TOTAL_FES: Final[str] = (
    "the total number of objective function evaluations (FEs) that were "
    "performed during the run.")
#: the total consumed time in milliseconds
DESC_TOTAL_TIME_MILLIS: Final[str] = (
    "the clock time in milliseconds that has passed between the begin of the "
    "run and the end of the run.")
#: the description of the goal objective value
DESC_GOAL_F: Final[str] = (
    "the goal objective value. A run will stop as soon as a solution was"
    "discovered which has an objective value less than or equal to "
    f"{KEY_GOAL_F}. In other words, as soon as {KEY_BEST_F} reaches or dips "
    f"under {KEY_GOAL_F}, the algorithm will stop. If {KEY_GOAL_F} is not "
    "reached, the run will continue until other budget limits are exhausted. "
    "If a lower bound for the objective function is known, this is often used"
    " as a goal objective value. If o goal objective value is specified, this"
    " field is empty.")
#: a description of the budget as the maximum objective function evaluation
DESC_MAX_FES: Final[str] = (
    "the maximum number of permissible FEs per run. As soon as this limit is "
    f"reached, the run will stop. In other words, {KEY_TOTAL_FES} will never "
    f"be more than {KEY_MAX_FES}. A run may stop earlier if some other "
    "termination criterion is reached, but never later.")
#: a description of the budget in terms of maximum runtime
DESC_MAX_TIME_MILLIS: Final[str] = (
    "the maximum number of milliseconds of clock time that a run is permitted"
    " to use as computational budget before being terminated. This limit is "
    "more of a soft limit, as we cannot physically stop a run at arbitrary "
    "points without causing mayhem. Thus, it may be that some runs consume "
    "slightly more runtime than this limit. But the rule is that the "
    "algorithm gets told to stop (via should_terminate() becoming True) as "
    f"soon as this time has elapsed. But generally, {KEY_TOTAL_TIME_MILLIS}<="
    f"{KEY_MAX_TIME_MILLIS} approximately holds.")


@dataclass(frozen=True, init=False, order=False, eq=False)
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
                 objective: str,
                 encoding: str | None,
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
        :param objective: the name of the objective function
        :param encoding: the name of the encoding that was used, if any, or
            `None` if no encoding was used
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
        super().__init__(algorithm, instance, objective, encoding, rand_seed)
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

    def _tuple(self) -> tuple[Any, ...]:
        """
        Get the tuple representation of this object used in comparisons.

        :return: the comparison-relevant data of this object in a tuple
        """
        return (self.__class__.__name__,
                "" if self.algorithm is None else self.algorithm,
                "" if self.instance is None else self.instance,
                "" if self.objective is None else self.objective,
                "" if self.encoding is None else self.encoding,
                1, self.rand_seed, "", "",
                inf if self.goal_f is None else self.goal_f,
                inf if self.max_fes is None else self.max_fes,
                inf if self.max_time_millis is None else self.max_time_millis,
                self.best_f, self.last_improvement_fe,
                self.last_improvement_time_millis, self.total_fes,
                self.total_time_millis)

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
        return Path(base_dir).resolve_inside(
            self.algorithm).resolve_inside(self.instance).resolve_inside(
            sanitize_names([self.algorithm, self.instance,
                            hex(self.rand_seed)]) + FILE_SUFFIX)

    def get_best_f(self) -> int | float:
        """
        Get the best objective value reached.

        :returns: the best objective value reached
        """
        if not isinstance(self, EndResult):
            raise type_error(self, "self", EndResult)
        return self.best_f

    def get_last_improvement_fe(self) -> int:
        """
        Get the index of the function evaluation when `best_f` was reached.

        :returns: the index of the function evaluation when `best_f` was
            reached
        """
        if not isinstance(self, EndResult):
            raise type_error(self, "self", EndResult)
        return self.last_improvement_fe

    def get_last_improvement_time_millis(self) -> int:
        """
        Get the milliseconds when `best_f` was reached.

        :returns: the milliseconds when `best_f` was reached
        """
        if not isinstance(self, EndResult):
            raise type_error(self, "self", EndResult)
        return self.last_improvement_time_millis

    def get_total_fes(self) -> int:
        """
        Get the total number of performed FEs.

        :returns: the total number of performed FEs
        """
        if not isinstance(self, EndResult):
            raise type_error(self, "self", EndResult)
        return self.total_fes

    def get_total_time_millis(self) -> int:
        """
        Get the total time consumed by the run.

        :returns: the total time consumed by the run
        """
        if not isinstance(self, EndResult):
            raise type_error(self, "self", EndResult)
        return self.total_time_millis

    def get_goal_f(self) -> int | float | None:
        """
        Get the goal objective value, if any.

        :returns: the goal objective value, if any
        """
        if not isinstance(self, EndResult):
            raise type_error(self, "self", EndResult)
        return self.goal_f

    def get_max_fes(self) -> int | None:
        """
        Get the maximum number of FEs permissible.

        :returns: the maximum number of FEs permissible
        """
        if not isinstance(self, EndResult):
            raise type_error(self, "self", EndResult)
        return self.max_fes

    def get_max_time_millis(self) -> int | None:
        """
        Get the maximum permissible milliseconds permitted.

        :returns: the maximum permissible milliseconds permitted
        """
        if not isinstance(self, EndResult):
            raise type_error(self, "self", EndResult)
        return self.max_time_millis

    def get_normalized_best_f(self) -> int | float | None:
        """
        Get the normalized f.

        :returns: the normalized f
        """
        g: Final[int | float | None] = EndResult.get_goal_f(self)
        if (g is None) or (g <= 0):
            return None
        return try_float_div(self.best_f - g, g)

    def get_scaled_best_f(self) -> int | float | None:
        """
        Get the normalized f.

        :returns: the normalized f
        """
        g: Final[int | float | None] = EndResult.get_goal_f(self)
        if (g is None) or (g <= 0):
            return None
        return try_float_div(self.best_f, g)

    def get_fes_per_time_milli(self) -> int | float:
        """
        Get the fes per time milliseconds.

        :returns: the fes per time milliseconds
        """
        return try_int_div(EndResult.get_total_fes(self), max(
            1, EndResult.get_total_time_millis(self)))


#: A set of getters for accessing variables of the end result
__PROPERTIES: Final[Callable[[str], Callable[[
    EndResult], int | float | None]]] = {
    KEY_LAST_IMPROVEMENT_FE: EndResult.get_last_improvement_fe,
    "last improvement FE": EndResult.get_last_improvement_fe,
    KEY_LAST_IMPROVEMENT_TIME_MILLIS:
        EndResult.get_last_improvement_time_millis,
    "last improvement ms": EndResult.get_last_improvement_time_millis,
    KEY_TOTAL_FES: EndResult.get_total_fes,
    "fes": EndResult.get_total_fes,
    KEY_TOTAL_TIME_MILLIS: EndResult.get_total_time_millis,
    "ms": EndResult.get_total_time_millis,
    KEY_GOAL_F: EndResult.get_goal_f,
    F_NAME_RAW: EndResult.get_best_f,
    KEY_BEST_F: EndResult.get_best_f,
    "f": EndResult.get_best_f,
    F_NAME_SCALED: EndResult.get_scaled_best_f,
    "bestFscaled": EndResult.get_scaled_best_f,
    F_NAME_NORMALIZED: EndResult.get_normalized_best_f,
    "bestFnormalized": EndResult.get_normalized_best_f,
    KEY_MAX_FES: EndResult.get_max_fes,
    "budgetFEs": EndResult.get_max_fes,
    KEY_MAX_TIME_MILLIS: EndResult.get_max_time_millis,
    "budgetMS": EndResult.get_max_time_millis,
    "fesPerTimeMilli": EndResult.get_fes_per_time_milli,
}.get


def getter(dimension: str) -> Callable[[EndResult], int | float | None]:
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
    result: Callable[[EndResult], int | float] | None = __PROPERTIES(
        str.strip(dimension))
    if result is None:
        raise ValueError(f"Unknown EndResult dimension {dimension!r}.")
    return result


def from_logs(
        path: str, consumer: Callable[[EndResult], Any],
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
        __InnerProgressLogParser(
            max_fes, max_time_millis, goal_f, consumer).parse(path)
    else:
        __InnerLogParser(consumer).parse(path)


def to_csv(results: Iterable[EndResult], file: str) -> Path:
    """
    Write a sequence of end results to a file in CSV format.

    :param results: the end results
    :param file: the path
    :return: the path of the file that was written
    """
    path: Final[Path] = Path(file)
    logger(f"Writing end results to CSV file {path!r}.")
    path.ensure_parent_dir_exists()
    with path.open_for_write() as wt:
        csv_write(data=sorted(results),
                  consumer=line_writer(wt),
                  setup=CsvWriter().setup,
                  get_column_titles=CsvWriter.get_column_titles,
                  get_row=CsvWriter.get_row,
                  get_header_comments=CsvWriter.get_header_comments,
                  get_footer_comments=CsvWriter.get_footer_comments)
    logger(f"Done writing end results to CSV file {path!r}.")
    return path


def from_csv(file: str, consumer: Callable[[EndResult], Any],
             filterer: Callable[[EndResult], bool]
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
    path: Final[Path] = file_path(file)
    logger(f"Now reading CSV file {path!r}.")

    def __cons(r: EndResult, __c=consumer, __f=filterer) -> None:
        """Consume a record."""
        if __f(r):
            __c(r)

    with path.open_for_read() as rd:
        csv_read(rows=rd,
                 setup=CsvReader,
                 parse_row=CsvReader.parse_row,
                 consumer=__cons)

    logger(f"Done reading CSV file {path!r}.")


class CsvWriter:
    """A class for CSV writing of :class:`EndResult`."""

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
        #: do we need the encoding?
        self.__needs_encoding: bool = False
        #: do we need the max FEs?
        self.__needs_max_fes: bool = False
        #: do we need the max millis?
        self.__needs_max_ms: bool = False
        #: do we need the goal F?
        self.__needs_goal_f: bool = False

    def setup(self, data: Iterable[EndResult]) -> "CsvWriter":
        """
        Set up this csv writer based on existing data.

        :param data: the data to setup with
        :returns: this writer
        """
        if self.__setup:
            raise ValueError(
                "EndResults CsvWriter has already been set up.")
        self.__setup = True

        no_encoding: bool = True
        no_max_fes: bool = True
        no_max_ms: bool = True
        no_goal_f: bool = True
        check: int = 4
        for er in data:
            if no_encoding and (er.encoding is not None):
                no_encoding = False
                self.__needs_encoding = True
                check -= 1
                if check <= 0:
                    return self
            if no_max_fes and (er.max_fes is not None):
                self.__needs_max_fes = True
                no_max_fes = False
                check -= 1
                if check <= 0:
                    return self
            if no_max_ms and (er.max_time_millis is not None):
                self.__needs_max_ms = True
                no_max_ms = False
                check -= 1
                if check <= 0:
                    return self
            if no_goal_f and (er.goal_f is not None) and (
                    isfinite(er.goal_f)):
                self.__needs_goal_f = True
                no_goal_f = False
                check -= 1
                if check <= 0:
                    return self
        return self

    def get_column_titles(self, dest: Callable[[str], None]) -> None:
        """
        Get the column titles.

        :param dest: the destination string consumer
        """
        p: Final[str] = self.scope
        dest(csv_scope(p, KEY_ALGORITHM))
        dest(csv_scope(p, KEY_INSTANCE))
        dest(csv_scope(p, KEY_OBJECTIVE_FUNCTION))
        if self.__needs_encoding:
            dest(csv_scope(p, KEY_ENCODING))
        dest(csv_scope(p, KEY_RAND_SEED))
        dest(csv_scope(p, KEY_BEST_F))
        dest(csv_scope(p, KEY_LAST_IMPROVEMENT_FE))
        dest(csv_scope(p, KEY_LAST_IMPROVEMENT_TIME_MILLIS))
        dest(csv_scope(p, KEY_TOTAL_FES))
        dest(csv_scope(p, KEY_TOTAL_TIME_MILLIS))

        if self.__needs_goal_f:
            dest(csv_scope(p, KEY_GOAL_F))
        if self.__needs_max_fes:
            dest(csv_scope(p, KEY_MAX_FES))
        if self.__needs_max_ms:
            dest(csv_scope(p, KEY_MAX_TIME_MILLIS))

    def get_row(self, data: EndResult,
                dest: Callable[[str], None]) -> None:
        """
        Render a single end result record to a CSV row.

        :param data: the end result record
        :param dest: the string consumer
        """
        dest(data.algorithm)
        dest(data.instance)
        dest(data.objective)
        if self.__needs_encoding:
            dest(data.encoding if data.encoding else "")
        dest(hex(data.rand_seed))
        dest(num_to_str(data.best_f))
        dest(str(data.last_improvement_fe))
        dest(str(data.last_improvement_time_millis))
        dest(str(data.total_fes))
        dest(str(data.total_time_millis))
        if self.__needs_goal_f:
            dest(num_or_none_to_str(data.goal_f))
        if self.__needs_max_fes:
            dest(int_or_none_to_str(data.max_fes))
        if self.__needs_max_ms:
            dest(int_or_none_to_str(data.max_time_millis))

    def get_header_comments(self, dest: Callable[[str], None]) -> None:
        """
        Get any possible header comments.

        :param dest: the destination
        """
        dest("Experiment End Results")
        dest("See the description at the bottom of the file.")

    def get_footer_comments(self, dest: Callable[[str], None]) -> None:
        """
        Get any possible footer comments.

        :param dest: the destination
        """
        dest("")
        scope: Final[str | None] = self.scope
        dest("Records describing the end results of single runs ("
             "single executions) of algorithms applied to optimization "
             "problems.")
        dest("Each run is characterized by an algorithm setup, a problem "
             "instance, and a random seed.")
        if scope:
            dest("All end result records start with prefix "
                 f"{scope}{SCOPE_SEPARATOR}.")
        dest(f"{csv_scope(scope, KEY_ALGORITHM)}: {DESC_ALGORITHM}")
        dest(f"{csv_scope(scope, KEY_INSTANCE)}: {DESC_INSTANCE}")
        dest(f"{csv_scope(scope, KEY_OBJECTIVE_FUNCTION)}:"
             f" {DESC_OBJECTIVE_FUNCTION}")
        if self.__needs_encoding:
            dest(f"{csv_scope(scope, KEY_ENCODING)}: {DESC_ENCODING}")
        dest(f"{csv_scope(scope, KEY_RAND_SEED)}: {DESC_RAND_SEED}")
        dest(f"{csv_scope(scope, KEY_BEST_F)}: {DESC_BEST_F}")
        dest(f"{csv_scope(scope, KEY_LAST_IMPROVEMENT_FE)}: "
             f"{DESC_LAST_IMPROVEMENT_FE}")
        dest(f"{csv_scope(scope, KEY_LAST_IMPROVEMENT_TIME_MILLIS)}: "
             f"{DESC_LAST_IMPROVEMENT_TIME_MILLIS}")
        dest(f"{csv_scope(scope, KEY_TOTAL_FES)}: {DESC_TOTAL_FES}")
        dest(f"{csv_scope(scope, KEY_TOTAL_TIME_MILLIS)}: "
             f"{DESC_TOTAL_TIME_MILLIS}")
        if self.__needs_goal_f:
            dest(f"{csv_scope(scope, KEY_GOAL_F)}: {DESC_GOAL_F}")
        if self.__needs_max_fes:
            dest(f"{csv_scope(scope, KEY_MAX_FES)}: {DESC_MAX_FES}")
        if self.__needs_max_ms:
            dest(f"{csv_scope(scope, KEY_MAX_TIME_MILLIS)}: "
                 f"{DESC_MAX_TIME_MILLIS}")
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
        self.__idx_algorithm: Final[int] = csv_column(columns, KEY_ALGORITHM)
        #: the index of the instance column, if any
        self.__idx_instance: Final[int] = csv_column(columns, KEY_INSTANCE)
        #: the index of the objective function column, if any
        self.__idx_objective: Final[int] = csv_column(
            columns, KEY_OBJECTIVE_FUNCTION)
        #: the index of the encoding column, if any
        self.__idx_encoding = csv_column_or_none(columns, KEY_ENCODING)

        #: the index of the random seed column
        self.__idx_seed: Final[int] = csv_column(columns, KEY_RAND_SEED)
        #: the column with the last improvement FE
        self.__idx_li_fe: Final[int] = csv_column(
            columns, KEY_LAST_IMPROVEMENT_FE)
        #: the column with the last improvement time milliseconds
        self.__idx_li_ms: Final[int] = csv_column(
            columns, KEY_LAST_IMPROVEMENT_TIME_MILLIS)
        #: the column with the best obtained objective value
        self.__idx_best_f: Final[int] = csv_column(columns, KEY_BEST_F)
        #: the column with the total time in FEs
        self.__idx_tt_fe: Final[int] = csv_column(columns, KEY_TOTAL_FES)
        #: the column with the total time in milliseconds
        self.__idx_tt_ms: Final[int] = csv_column(
            columns, KEY_TOTAL_TIME_MILLIS)

        #: the column with the goal objective value, if any
        self.__idx_goal_f: Final[int | None] = csv_column_or_none(
            columns, KEY_GOAL_F)
        #: the column with the maximum FEs, if any such budget constraint was
        #: defined
        self.__idx_max_fes: Final[int | None] = csv_column_or_none(
            columns, KEY_MAX_FES)
        #: the column with the maximum runtime in milliseconds, if any such
        #: budget constraint was defined
        self.__idx_max_ms: Final[int | None] = csv_column_or_none(
            columns, KEY_MAX_TIME_MILLIS)

    def parse_row(self, data: list[str]) -> EndResult:
        """
        Parse a row of data.

        :param data: the data row
        :return: the end result statistics
        """
        return EndResult(
            data[self.__idx_algorithm],  # algorithm
            data[self.__idx_instance],  # instance
            data[self.__idx_objective],  # objective
            csv_str_or_none(data, self.__idx_encoding),  # encoding
            int((data[self.__idx_seed])[2:], 16),  # rand seed
            str_to_num(data[self.__idx_best_f]),  # best_f
            int(data[self.__idx_li_fe]),  # last_improvement_fe
            int(data[self.__idx_li_ms]),  # last_improvement_time_millis
            int(data[self.__idx_tt_fe]),  # total_fes
            int(data[self.__idx_tt_ms]),  # total_time_millis
            csv_val_or_none(data, self.__idx_goal_f, str_to_num),
            csv_val_or_none(data, self.__idx_max_fes, int),  # max_fes
            csv_val_or_none(data, self.__idx_max_ms, int))  # max_time_ms


class __InnerLogParser(SetupAndStateParser):
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
                                  self.objective,
                                  self.encoding,
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


class __InnerProgressLogParser(SetupAndStateParser):
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
            objective=self.objective,
            encoding=self.encoding,
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
            f: int | float = str_to_num(values[f_col])
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
    parser: Final[argparse.ArgumentParser] = moptipy_argparser(
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
             "under which to search for log files", type=Path)
    parser.add_argument(
        "dest", help="the path to the end results CSV file to be created",
        type=Path, nargs="?", default="./evaluation/end_results.txt")
    parser.add_argument(
        "--maxFEs", help="the maximum permitted FEs",
        type=int, nargs="?", default=None)
    parser.add_argument(
        "--maxTime", help="the maximum permitted time in milliseconds",
        type=int, nargs="?", default=None)
    parser.add_argument(
        "--goalF", help="the goal objective value",
        type=str_to_num, nargs="?", default=None)
    args: Final[argparse.Namespace] = parser.parse_args()

    end_results: Final[list[EndResult]] = []
    from_logs(args.source, end_results.append,
              args.maxFEs, args.maxTime, args.goalF)
    to_csv(end_results, args.dest)
