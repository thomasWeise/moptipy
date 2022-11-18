"""Record for EndResult as well as parsing, serialization, and parsing."""
import os.path
import sys
from dataclasses import dataclass
from math import inf, isfinite
from typing import Any, Callable, Final, Iterable

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
    SECTION_FINAL_STATE,
    SECTION_SETUP,
)
from moptipy.evaluation._utils import (
    _FULL_KEY_GOAL_F,
    _FULL_KEY_MAX_FES,
    _FULL_KEY_MAX_TIME_MILLIS,
    _FULL_KEY_RAND_SEED,
    _check_max_time_millis,
)
from moptipy.evaluation.base import (
    F_NAME_NORMALIZED,
    F_NAME_RAW,
    F_NAME_SCALED,
    PerRunData,
)
from moptipy.evaluation.log_parser import ExperimentParser
from moptipy.utils.console import logger
from moptipy.utils.help import help_screen
from moptipy.utils.logger import CSV_SEPARATOR, parse_key_values
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
from moptipy.utils.types import type_error

#: The internal CSV header
_HEADER = f"{KEY_ALGORITHM}{CSV_SEPARATOR}" \
          f"{KEY_INSTANCE}{CSV_SEPARATOR}" \
          f"{KEY_RAND_SEED}{CSV_SEPARATOR}" \
          f"{KEY_BEST_F}{CSV_SEPARATOR}" \
          f"{KEY_LAST_IMPROVEMENT_FE}{CSV_SEPARATOR}" \
          f"{KEY_LAST_IMPROVEMENT_TIME_MILLIS}" \
          f"{CSV_SEPARATOR}" \
          f"{KEY_TOTAL_FES}{CSV_SEPARATOR}" \
          f"{KEY_TOTAL_TIME_MILLIS}{CSV_SEPARATOR}" \
          f"{KEY_GOAL_F}{CSV_SEPARATOR}" \
          f"{KEY_MAX_FES}{CSV_SEPARATOR}" \
          f"{KEY_MAX_TIME_MILLIS}\n"


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

        if not isinstance(last_improvement_fe, int):
            raise type_error(last_improvement_fe, "last_improvement_fe", int)
        if last_improvement_fe <= 0:
            raise ValueError("last_improvement_fe must be > 0, "
                             f"but is {last_improvement_fe}.")
        object.__setattr__(self, "last_improvement_fe", last_improvement_fe)

        if not isinstance(last_improvement_time_millis, int):
            raise type_error(last_improvement_time_millis,
                             "last_improvement_time_millis", int)
        if last_improvement_fe < 0:
            raise ValueError("last_improvement_time_millis must be >= 0, "
                             f"but is {last_improvement_time_millis}.")
        object.__setattr__(self, "last_improvement_time_millis",
                           last_improvement_time_millis)

        if not isinstance(total_fes, int):
            raise type_error(total_fes, "total_fes", int)
        if last_improvement_fe > total_fes:
            raise ValueError("last_improvement_fe must be <= total_fes, "
                             f"but is {last_improvement_fe} vs. "
                             f"{total_fes}.")
        object.__setattr__(self, "total_fes", total_fes)

        if not isinstance(total_time_millis, int):
            raise type_error(total_time_millis,
                             "total_time_millis", int)
        if last_improvement_time_millis > total_time_millis:
            raise ValueError("last_improvement_fe must be <= total_fes, "
                             f"but is {last_improvement_time_millis} vs. "
                             f"{total_time_millis}.")
        object.__setattr__(self, "total_time_millis", total_time_millis)

        if goal_f is not None:
            if goal_f <= -inf:
                goal_f = None
            else:
                goal_f = try_int(goal_f)
        object.__setattr__(self, "goal_f", goal_f)

        if max_fes is not None:
            if not isinstance(max_fes, int):
                raise type_error(max_fes, "max_fes", int)
            if max_fes < total_fes:
                raise ValueError(f"max_fes ({max_fes}) must be >= total_fes "
                                 f"({total_fes}), but are not.")
        object.__setattr__(self, "max_fes", max_fes)

        if max_time_millis is not None:
            if not isinstance(max_time_millis, int):
                raise type_error(max_time_millis, "max_time_millis", int)
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
        raise ValueError(f"unknown dimension '{dimension}', "
                         f"should be one of {sorted(_GETTERS.keys())}.")

    @staticmethod
    def from_logs(path: str, consumer: Callable[["EndResult"], Any]) -> None:
        """
        Parse a given path and pass all end results found to the consumer.

        If `path` identifies a file with suffix `.txt`, then this file is
        parsed. The appropriate :class:`EndResult` is created and appended to
        the `collector`. If `path` identifies a directory, then this directory
        is parsed recursively for each log file found, one record is passed to
        the `consumer`. As `consumer`, you could pass any `callable` that
        accepts instances of :class:`EndResult`, e.g., the `append` method of
        a :class:`list`.

        :param path: the path to parse
        :param consumer: the consumer
        """
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
        logger(f"Writing end results to CSV file '{path}'.")
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

        logger(f"Done writing end results to CSV file '{path}'.")
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
        logger(f"Now reading CSV file '{path}'.")

        with path.open_for_read() as rd:
            header = rd.readlines(1)
            if (header is None) or (len(header) <= 0):
                raise ValueError(f"No line in file '{file}'.")
            if _HEADER != header[0]:
                raise ValueError(
                    f"Header '{header[0]}' in '{path}' should be {_HEADER}.")

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

        logger(f"Done reading CSV file '{path}'.")


class _InnerLogParser(ExperimentParser):
    """The internal log parser class."""

    def __init__(self, consumer: Callable[[EndResult], Any]):
        """
        Create the internal log parser.

        :param consumer: the consumer accepting the parsed data
        """
        super().__init__()
        if not callable(consumer):
            raise type_error(consumer, "consumer", call=True)
        self.__consumer: Final[Callable[["EndResult"], Any]] = consumer
        self.__total_fes: int | None = None
        self.__total_time_millis: int | None = None
        self.__best_f: int | float | None = None
        self.__last_improvement_fe: int | None = None
        self.__last_improvement_time_millis: int | None = None
        self.__goal_f: int | float | None = None
        self.__max_fes: int | None = None
        self.__max_time_millis: int | None = None
        self.__state: int = 0

    def start_file(self, path: Path) -> bool:
        if not super().start_file(path):
            return False

        if self.__state != 0:
            raise ValueError(f"Illegal state when trying to parse {path}.")

        return True

    def end_file(self) -> bool:
        if self.__state != 3:
            raise ValueError(
                "Illegal state, log file must have both a "
                f"{SECTION_FINAL_STATE} and a "
                f"{SECTION_SETUP} section.")

        if self.rand_seed is None:
            raise ValueError("rand_seed is missing.")
        if self.algorithm is None:
            raise ValueError("algorithm is missing.")
        if self.instance is None:
            raise ValueError("instance is missing.")
        if self.__total_fes is None:
            raise ValueError("total_fes is missing.")
        if self.__total_time_millis is None:
            raise ValueError("total_time_millis is missing.")
        if self.__best_f is None:
            raise ValueError("best_f is missing.")
        if self.__last_improvement_fe is None:
            raise ValueError("last_improvement_fe is missing.")
        if self.__last_improvement_time_millis is None:
            raise ValueError("last_improvement_time_millis is missing.")

        self.__consumer(EndResult(self.algorithm,
                                  self.instance,
                                  self.rand_seed,
                                  self.__best_f,
                                  self.__last_improvement_fe,
                                  self.__last_improvement_time_millis,
                                  self.__total_fes,
                                  self.__total_time_millis,
                                  self.__goal_f,
                                  self.__max_fes,
                                  self.__max_time_millis))

        self.__total_fes = None
        self.__total_time_millis = None
        self.__best_f = None
        self.__last_improvement_fe = None
        self.__last_improvement_time_millis = None
        self.__goal_f = None
        self.__max_fes = None
        self.__max_time_millis = None
        self.__state = 0
        return super().end_file()

    def start_section(self, title: str) -> bool:
        super().start_section(title)
        if title == SECTION_SETUP:
            if (self.__state & 1) != 0:
                raise ValueError(f"Already did section {title}.")
            self.__state |= 4
            return True
        if title == SECTION_FINAL_STATE:
            if (self.__state & 2) != 0:
                raise ValueError(f"Already did section {title}.")
            self.__state |= 8
            return True
        return False

    def lines(self, lines: list[str]) -> bool:
        data = parse_key_values(lines)
        if not isinstance(data, dict):
            raise type_error(data, "data", dict)

        if (self.__state & 4) != 0:
            if _FULL_KEY_GOAL_F in data:
                goal_f = data[_FULL_KEY_GOAL_F]
                if ("e" in goal_f) or ("E" in goal_f) or ("." in goal_f):
                    self.__goal_f = float(goal_f)
                elif goal_f == "-inf":
                    self.__goal_f = None
                else:
                    self.__goal_f = int(goal_f)
            else:
                self.__goal_f = None

            if _FULL_KEY_MAX_FES in data:
                self.__max_fes = int(data[_FULL_KEY_MAX_FES])
            if _FULL_KEY_MAX_TIME_MILLIS in data:
                self.__max_time_millis = \
                    int(data[_FULL_KEY_MAX_TIME_MILLIS])

            seed_check = int(data[_FULL_KEY_RAND_SEED])
            if seed_check != self.rand_seed:
                raise ValueError(
                    f"Found seed {seed_check} in log file, but file name "
                    f"indicates seed {self.rand_seed}.")

            self.__state = (self.__state | 1) & (~4)
            return self.__state != 3

        if (self.__state & 8) != 0:
            self.__total_fes = int(data[KEY_TOTAL_FES])
            self.__total_time_millis = \
                int(data[KEY_TOTAL_TIME_MILLIS])

            self.__best_f = str_to_intfloat(data[KEY_BEST_F])

            self.__last_improvement_fe = \
                int(data[KEY_LAST_IMPROVEMENT_FE])
            self.__last_improvement_time_millis = \
                int(data[KEY_LAST_IMPROVEMENT_TIME_MILLIS])

            self.__state = (self.__state | 2) & (~8)
            return self.__state != 3

        raise ValueError("Illegal state.")


# Run log files to end results if executed as script
if __name__ == "__main__":
    help_screen(
        "build end results-CSV from log files", __file__,
        "Convert log files obtained with moptipy to the end results "
        "CSV format.",
        [("source_dir", "the location of the moptipy data"),
         ("dest_file", "the path to which we want to write the CSV file.")])
    if len(sys.argv) != 3:
        raise ValueError("two command line arguments expected")

    end_results: Final[list[EndResult]] = []
    EndResult.from_logs(sys.argv[1], end_results.append)
    EndResult.to_csv(end_results, sys.argv[2])
