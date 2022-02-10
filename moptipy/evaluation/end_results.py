"""Record for EndResult as well as parsing, serialization, and parsing."""
from dataclasses import dataclass
from math import inf
from typing import Union, List, Final, Optional, Iterable, Callable, Any

from moptipy.api import logging
from moptipy.evaluation._utils import _FULL_KEY_RAND_SEED, _FULL_KEY_MAX_FES, \
    _FULL_KEY_GOAL_F, _FULL_KEY_MAX_TIME_MILLIS, _check_max_time_millis
from moptipy.evaluation.base import PerRunData
from moptipy.evaluation.log_parser import ExperimentParser
from moptipy.utils.log import logger
from moptipy.utils.logger import parse_key_values
from moptipy.utils.path import Path
from moptipy.utils.types import intfloatnone_to_str, intnone_to_str, \
    str_to_intfloat, str_to_intfloatnone, str_to_intnone, try_int, num_to_str

#: The internal CSV header
_HEADER = f"{logging.KEY_ALGORITHM}{logging.CSV_SEPARATOR}" \
          f"{logging.KEY_INSTANCE}{logging.CSV_SEPARATOR}" \
          f"{logging.KEY_RAND_SEED}{logging.CSV_SEPARATOR}" \
          f"{logging.KEY_BEST_F}{logging.CSV_SEPARATOR}" \
          f"{logging.KEY_LAST_IMPROVEMENT_FE}{logging.CSV_SEPARATOR}" \
          f"{logging.KEY_LAST_IMPROVEMENT_TIME_MILLIS}" \
          f"{logging.CSV_SEPARATOR}" \
          f"{logging.KEY_TOTAL_FES}{logging.CSV_SEPARATOR}" \
          f"{logging.KEY_TOTAL_TIME_MILLIS}{logging.CSV_SEPARATOR}" \
          f"{logging.KEY_MAX_FES}{logging.CSV_SEPARATOR}" \
          f"{logging.KEY_MAX_TIME_MILLIS}{logging.CSV_SEPARATOR}" \
          f"{logging.KEY_GOAL_F}\n"


@dataclass(frozen=True, init=False, order=True)
class EndResult(PerRunData):
    """
    An immutable end result record of one run of one algorithm on one problem.

    This record provides the information of the outcome of one application of
    one algorithm to one problem instance in an immutable way.
    """

    #: The best objective value encountered.
    best_f: Union[int, float]

    #: The index of the function evaluation when best_f was reached.
    last_improvement_fe: int

    #: The time when best_f was reached.
    last_improvement_time_millis: int

    #: The total number of performed FEs.
    total_fes: int

    #: The total time consumed by the run.
    total_time_millis: int

    #: The goal objective value if provided
    goal_f: Union[int, float, None]

    #: The (optional) maximum permitted FEs.
    max_fes: Optional[int]

    #: The (optional) maximum runtime.
    max_time_millis: Optional[int]

    def __init__(self,
                 algorithm: str,
                 instance: str,
                 rand_seed: int,
                 best_f: Union[int, float],
                 last_improvement_fe: int,
                 last_improvement_time_millis: int,
                 total_fes: int,
                 total_time_millis: int,
                 goal_f: Union[int, float, None],
                 max_fes: Optional[int],
                 max_time_millis: Optional[int]):
        """
        Create a consistent instance of :class:`EndResult`.

        :param str algorithm: the algorithm name
        :param str instance: the instance name
        :param int rand_seed: the random seed
        :param Union[int,float] best_f: the best reached objective value
        :param int last_improvement_fe: the FE when best_f was reached
        :param int last_improvement_time_millis: the time when best_f
            was reached
        :param int total_fes: the total FEs
        :param int total_time_millis: the total runtime
        :param Union[int, float, None] goal_f: the goal objective value, if
            provide
        :param Optional[int] max_fes: the optional maximum FEs
        :param Optional[int] max_time_millis: the optional maximum runtime

        :raises TypeError: if any parameter has a wrong type
        :raises ValueError: if the parameter values are inconsistent
        """
        super().__init__(algorithm, instance, rand_seed)
        object.__setattr__(self, "best_f", try_int(best_f))

        if not isinstance(last_improvement_fe, int):
            raise TypeError("last_improvement_fe must be int, "
                            f"but is {type(last_improvement_fe)}.")
        if last_improvement_fe <= 0:
            raise ValueError("last_improvement_fe must be > 0, "
                             f"but is {last_improvement_fe}.")
        object.__setattr__(self, "last_improvement_fe", last_improvement_fe)

        if not isinstance(last_improvement_time_millis, int):
            raise TypeError("last_improvement_time_millis must be int, "
                            f"but is {type(last_improvement_time_millis)}.")
        if last_improvement_fe < 0:
            raise ValueError("last_improvement_time_millis must be >= 0, "
                             f"but is {last_improvement_time_millis}.")
        object.__setattr__(self, "last_improvement_time_millis",
                           last_improvement_time_millis)

        if not isinstance(total_fes, int):
            raise TypeError("total_fes must be int, "
                            f"but is {type(total_fes)}.")
        if last_improvement_fe > total_fes:
            raise ValueError("last_improvement_fe must be <= total_fes, "
                             f"but is {last_improvement_fe} vs. "
                             f"{total_fes}.")
        object.__setattr__(self, "total_fes", total_fes)

        if not isinstance(total_time_millis, int):
            raise TypeError("total_time_millis must be int, "
                            f"but is {type(total_time_millis)}.")
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
                raise TypeError(
                    f"max_fes must be int but are {type(max_fes)}.")
            if max_fes < total_fes:
                raise ValueError(f"max_fes ({max_fes}) must be >= total_fes "
                                 f"({total_fes}), but are not.")
        object.__setattr__(self, "max_fes", max_fes)

        if max_time_millis is not None:
            if not isinstance(max_time_millis, int):
                raise TypeError("max_time_millis must be int but "
                                f"are {type(max_time_millis)}.")
            _check_max_time_millis(max_time_millis,
                                   total_fes,
                                   total_time_millis)
        object.__setattr__(self, "max_time_millis", max_time_millis)

    def success(self) -> bool:
        """
        Check if a run is successful.

        This method returns `True` if and only if `goal_f` is defined and
        `best_f <= goal_f` (and `False` otherwise).

        :return: True if `best_f<=goal_f`
        :rtype: bool
        """
        return False if self.goal_f is None else self.best_f <= self.goal_f

    @staticmethod
    def from_logs(path: str, consumer: Callable[['EndResult'], Any]) -> None:
        """
        Parse a given path and pass all end results found to the consumer.

        If `path` identifies a file with suffix `.txt`, then this file is
        parsed. The appropriate :class:`EndResult` is created and appended to
        the `collector`. If `path` identifies a directory, then this directory
        is parsed recursively for each log file found, one record is passed to
        the `consumer`. As `consumer`, you could pass any `callable` that
        accepts instances of :class:`EndResult`, e.g., the `append` method of
        a list.

        :param str path: the path to parse
        :param Callable[['EndResult'], Any] consumer: the consumer
        """
        _InnerLogParser(consumer).parse(path)

    @staticmethod
    def to_csv(results: Iterable['EndResult'], file: str) -> Path:
        """
        Write a sequence of end results to a file in CSV format.

        :param Iterable[EndResult] results: the end results
        :param str file: the path
        :return: the path of the file that was written
        :rtype: Path
        """
        path: Final[Path] = Path.path(file)
        logger(f"Writing end results to CSV file '{path}'.")

        with path.open_for_write() as out:
            out.write(_HEADER)
            for e in results:
                out.write(
                    f"{e.algorithm}{logging.CSV_SEPARATOR}"
                    f"{e.instance}{logging.CSV_SEPARATOR}"
                    f"{hex(e.rand_seed)}{logging.CSV_SEPARATOR}"
                    f"{num_to_str(e.best_f)}{logging.CSV_SEPARATOR}"
                    f"{e.last_improvement_fe}{logging.CSV_SEPARATOR}"
                    f"{e.last_improvement_time_millis}{logging.CSV_SEPARATOR}"
                    f"{e.total_fes}{logging.CSV_SEPARATOR}"
                    f"{e.total_time_millis}{logging.CSV_SEPARATOR}"
                    f"{intfloatnone_to_str(e.goal_f)}{logging.CSV_SEPARATOR}"
                    f"{intnone_to_str(e.max_fes)}{logging.CSV_SEPARATOR}"
                    f"{intnone_to_str(e.max_time_millis)}\n")

        logger(f"Done writing end results to CSV file '{path}'.")
        return path

    @staticmethod
    def from_csv(file: str, consumer: Callable[['EndResult'], Any],
                 filterer: Callable = lambda x: True) -> None:
        """
        Parse a given CSV file to get :class:`EndResult` Records.

        :param str file: the path to parse
        :param Callable[['EndResult'], Any] consumer: the collector
        :param Callable filterer: an optional filter function
        """
        if not callable(consumer):
            raise TypeError(
                f"Consumer must be callable, but is {type(consumer)}.")
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
                    splt = line.strip().split(logging.CSV_SEPARATOR)
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

        :param Callable[['EndResult'], Any] consumer: the consumer accepting
            the parsed data
        """
        super().__init__()
        if not callable(consumer):
            raise TypeError(
                f"Consumer must be callable, but is {type(consumer)}.")
        self.__consumer: Final[Callable[['EndResult'], Any]] = consumer
        self.__total_fes: Optional[int] = None
        self.__total_time_millis: Optional[int] = None
        self.__best_f: Union[int, float, None] = None
        self.__last_improvement_fe: Optional[int] = None
        self.__last_improvement_time_millis: Optional[int] = None
        self.__goal_f: Union[int, float, None] = None
        self.__max_fes: Optional[int] = None
        self.__max_time_millis: Optional[int] = None
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
                f"{logging.SECTION_FINAL_STATE} and a "
                f"{logging.SECTION_SETUP} section.")

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
        if title == logging.SECTION_SETUP:
            if (self.__state & 1) != 0:
                raise ValueError(f"Already did section {title}.")
            self.__state |= 4
            return True
        if title == logging.SECTION_FINAL_STATE:
            if (self.__state & 2) != 0:
                raise ValueError(f"Already did section {title}.")
            self.__state |= 8
            return True
        return False

    def lines(self, lines: List[str]) -> bool:
        data = parse_key_values(lines)
        if not isinstance(data, dict):
            raise ValueError("Error when parsing data.")

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
            self.__total_fes = int(data[logging.KEY_TOTAL_FES])
            self.__total_time_millis = \
                int(data[logging.KEY_TOTAL_TIME_MILLIS])

            self.__best_f = str_to_intfloat(data[logging.KEY_BEST_F])

            self.__last_improvement_fe = \
                int(data[logging.KEY_LAST_IMPROVEMENT_FE])
            self.__last_improvement_time_millis = \
                int(data[logging.KEY_LAST_IMPROVEMENT_TIME_MILLIS])

            self.__state = (self.__state | 2) & (~8)
            return self.__state != 3

        raise ValueError("Illegal state.")
