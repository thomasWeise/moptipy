"""Progress data over a run."""
from dataclasses import dataclass
from math import inf, isfinite
from typing import Any, Callable, Final

import numpy as np

from moptipy.api import logging
from moptipy.evaluation._utils import _FULL_KEY_GOAL_F, _FULL_KEY_RAND_SEED
from moptipy.evaluation.base import (
    F_NAME_RAW,
    F_NAME_SCALED,
    TIME_UNIT_FES,
    TIME_UNIT_MILLIS,
    PerRunData,
    check_f_name,
    check_time_unit,
)
from moptipy.evaluation.log_parser import ExperimentParser
from moptipy.utils.console import logger
from moptipy.utils.logger import (
    COMMENT_CHAR,
    CSV_SEPARATOR,
    KEY_VALUE_SEPARATOR,
    parse_key_values,
)
from moptipy.utils.nputils import is_all_finite, is_np_float, is_np_int
from moptipy.utils.path import Path
from moptipy.utils.strings import num_to_str, str_to_intfloat
from moptipy.utils.types import type_error


@dataclass(frozen=True, init=False, order=True)
class Progress(PerRunData):
    """An immutable record of progress information over a single run."""

    #: The time axis data.
    time: np.ndarray

    #: The unit of the time axis.
    time_unit: str

    #: The objective value data.
    f: np.ndarray

    #: the name of the objective value axis.
    f_name: str

    #: the standard value of the objective dimension.
    #: If :attr:`f_name` is `F_NAME_SCALED` or `F_NAME_NORMALIZED`.
    #: then this value has been used to normalize the data.
    f_standard: int | float | None

    def __init__(self,
                 algorithm: str,
                 instance: str,
                 rand_seed: int,
                 time: np.ndarray,
                 time_unit: str,
                 f: np.ndarray,
                 f_name: str,
                 f_standard: int | float | None = None,
                 only_improvements: bool = True):
        """
        Create a consistent instance of :class:`EndResult`.

        :param algorithm: the algorithm name
        :param instance: the instance name
        :param rand_seed: the random seed
        :param time: the time axis data
        :param time_unit: the unit of the time axis
        :param f: the objective value axis data
        :param f_name: the name of the objective value axis data
        :param f_standard: the value used to standardize of the objective
            value dimension
        :param only_improvements: enforce that f-values should be
            improving and time values increasing
        """
        super().__init__(algorithm, instance, rand_seed)

        if not isinstance(time, np.ndarray):
            raise type_error(time, "time data", np.ndarray)
        time.flags.writeable = False
        if len(time.shape) != 1:
            raise ValueError("time array must be one-dimensional, but "
                             f"has shape {time.shape}.")
        if not is_np_int(time.dtype):
            raise TypeError("time data must be integer-valued, "
                            f"but has type {time.dtype}.")
        tl = time.size
        if tl <= 0:
            raise ValueError("time data must not be empty.")
        if tl > 1:
            if only_improvements:
                if np.any(time[1:] <= time[:-1]):
                    raise ValueError("time data must be strictly increasing,"
                                     f"but encountered {time}.")
            else:
                if np.any(time[1:] < time[:-1]):
                    raise ValueError("time data must be monotonously"
                                     f"increasing, but encountered {time}.")

        object.__setattr__(self, "time", time)
        object.__setattr__(self, "time_unit", check_time_unit(time_unit))

        mintime = 1 if time_unit == TIME_UNIT_FES else 0
        if any(time < mintime):
            raise ValueError(f"No time value can be less than {mintime} if"
                             f" time unit is {time_unit}.")

        if not isinstance(f, np.ndarray):
            raise type_error(f, "f data", np.ndarray)
        f.flags.writeable = False
        if len(f.shape) != 1:
            raise ValueError(
                f"f array must be one-dimensional, but has shape {f.shape}.")
        if is_np_float(f.dtype):
            if not is_all_finite(f):
                raise ValueError("f must be all finite.")
        elif not is_np_int(f.dtype):
            raise TypeError("f data must be integer- or float valued, but"
                            f" encountered an {type(f)} of {f.dtype}.")
        fl = f.size
        if fl <= 0:
            raise ValueError("f data must not be empty.")
        if fl != tl:
            raise ValueError(f"Length {fl} of f data and length {tl} of "
                             "time data must be the same.")
        if not isinstance(only_improvements, bool):
            raise type_error(only_improvements, "only_improvements", bool)
        if only_improvements and (fl > 1):
            if np.any(f[1:-1] >= f[:-2]):
                raise ValueError(
                    "f data must be strictly decreasing, with "
                    "only the entry being permitted as exception.")
            if f[-1] > f[-2]:
                raise ValueError(f"last f-value ({f[-1]}) cannot be greater"
                                 f"than second-to-last ({f[-2]}).")
        object.__setattr__(self, "f", f)
        object.__setattr__(self, "f_name", check_f_name(f_name))

        if (f_name != F_NAME_RAW) and (f_standard is None):
            raise ValueError(f"If f_name is {F_NAME_RAW}, "
                             f"then f_standard cannot be {f_standard}.")
        if f_standard is not None:
            if isinstance(f_standard, float):
                if not isfinite(f_standard):
                    raise ValueError(f"f_standard cannot be {f_standard}.")
            elif not isinstance(f_standard, int):
                raise type_error(f_standard, "f_standard", (int, float))
        object.__setattr__(self, "f_standard", f_standard)

    @staticmethod
    def from_logs(path: str,
                  consumer: Callable[["Progress"], Any],
                  time_unit: str = TIME_UNIT_FES,
                  f_name: str = F_NAME_RAW,
                  f_standard: dict[str, int | float] | None = None,
                  only_improvements: bool = True) -> None:
        """
        Parse a given path and pass all progress data found to the consumer.

        If `path` identifies a file with suffix `.txt`, then this file is
        parsed. The appropriate :class:`Progress` is created and appended to
        the `collector`. If `path` identifies a directory, then this directory
        is parsed recursively for each log file found, one record is passed to
        the `consumer`. The `consumer` is simply a callable function. You could
        pass in the `append` method of a :class:`list`.

        :param path: the path to parse
        :param consumer: the consumer, can be the `append` method of a
            :class:`list`
        :param time_unit: the time unit
        :param f_name: the objective name
        :param f_standard: a dictionary mapping instances to standard values
        :param only_improvements: enforce that f-values should be improving and
            time values increasing
        """
        _InnerLogParser(time_unit, f_name, consumer, f_standard,
                        only_improvements).parse(path)

    def to_csv(self, file: str,
               put_header: bool = True) -> str:
        """
        Store a :class:`Progress` record in a CSV file.

        :param file: the file to generate
        :param put_header: should we put a header with meta-data?
        :return: the fully resolved file name
        """
        path: Final[Path] = Path.path(file)
        logger(f"Writing progress object to CSV file '{path}'.")

        with path.open_for_write() as out:
            sep: Final[str] = CSV_SEPARATOR
            if put_header:
                kv: Final[str] = KEY_VALUE_SEPARATOR
                cmt: Final[str] = COMMENT_CHAR
                out.write(
                    f"{cmt} {logging.KEY_ALGORITHM}{kv}{self.algorithm}\n")
                out.write(
                    f"{cmt} {logging.KEY_INSTANCE}{kv}{self.instance}\n")
                out.write(f"{cmt} {logging.KEY_RAND_SEED}{kv}"
                          f"{hex(self.rand_seed)}\n")
                if self.f_standard is not None:
                    out.write(
                        f"{cmt} {logging.KEY_GOAL_F}{kv}{self.f_standard}\n")
            out.write(f"{self.time_unit}{sep}{self.f_name}\n")
            for i, t in enumerate(self.time):
                out.write(f"{t}{sep}{num_to_str(self.f[i])}\n")

        logger(f"Done writing progress object to CSV file '{path}'.")

        path.enforce_file()
        return path


class _InnerLogParser(ExperimentParser):
    """The internal log parser class."""

    # States are OR combinations of the following:
    # --> 0 == initial and end state
    #  1: SECTION_SETUP is done
    #  2: SECTION_FINAL_STATE is done
    #  4: SECTION_PROGRESS is done
    # --> 7 == all necessary data should have been collected
    #  8: SECTION_SETUP is in progress
    # 16: SECTION_FINAL_STATE is in progress
    # 32: SECTION_PROGRESS is in progress

    def __init__(self, time_unit: str, f_name: str,
                 consumer: Callable[[Progress], Any],
                 f_standard: dict[str, int | float] | None = None,
                 only_improvements: bool = True):
        """
        Create the internal log parser.

        :param time_unit: the time unit
        :param f_name: the objective name
        :param consumer: the consumer
        :param f_standard: a dictionary mapping instances to standard values
        :param only_improvements: enforce that f-values should be improving
            and time values increasing
        """
        super().__init__()
        if not callable(consumer):
            raise type_error(consumer, "consumer", call=True)
        self.__consumer: Final[Callable[[Progress], Any]] = consumer
        self.__time_unit = check_time_unit(time_unit)
        self.__f_name = check_f_name(f_name)
        self.__total_time: int | None = None
        self.__total_fes: int | None = None
        self.__last_fe: int | None = None
        self.__goal_f: int | float | None = None
        self.__t_collector: Final[list[int]] = []
        self.__f_collector: Final[list[int | float]] = []
        if not isinstance(only_improvements, bool):
            raise type_error(only_improvements, "only_improvements", bool)
        self.__only_improvements = only_improvements
        if (f_standard is not None) and (not isinstance(f_standard, dict)):
            raise type_error(f_standard, "f_standard", dict)
        self.__f_standard: Final[None | dict[str, int | float]] \
            = f_standard
        self.__state = 0

    def start_file(self, path: Path) -> bool:
        if not super().start_file(path):
            return False
        if self.__state != 0:
            raise ValueError(f"Illegal state when trying to parse {path}.")
        return True

    def end_file(self) -> bool:
        if self.__state != 7:
            raise ValueError(
                "Illegal state, log file must have a "
                f"{logging.SECTION_FINAL_STATE}, a "
                f"{logging.SECTION_SETUP}, and a "
                f"{logging.SECTION_PROGRESS} section.")

        if self.rand_seed is None:
            raise ValueError("rand_seed is missing.")
        if self.algorithm is None:
            raise ValueError("algorithm is missing.")
        if self.instance is None:
            raise ValueError("instance is missing.")
        if self.__total_time is None:
            raise ValueError("total time is missing.")
        if self.__total_fes is None:
            raise ValueError("total FEs are missing.")
        if not self.__f_collector:
            raise ValueError("f-collector cannot be empty.")
        if not self.__t_collector:
            raise ValueError("time-collector cannot be empty.")

        f_standard: int | float | None = None
        if (self.__f_standard is not None) and \
                (self.instance in self.__f_standard):
            f_standard = self.__f_standard[self.instance]
        if f_standard is None:
            f_standard = self.__goal_f
        if (self.__f_name != F_NAME_RAW) and (f_standard is None):
            raise ValueError(f"f_standard cannot be {f_standard} if f_name "
                             f"is {self.__f_name}.")

        tt = self.__total_time if (self.__time_unit == TIME_UNIT_MILLIS) \
            else self.__total_fes
        if tt < self.__t_collector[-1]:
            raise ValueError(
                f"Last time units {tt} inconsistent with last"
                f"recorded time unit {self.__t_collector[-1]}.")
        if self.__last_fe < self.__total_fes:
            if tt > self.__t_collector[-1]:
                self.__t_collector.append(tt)
                self.__f_collector.append(self.__f_collector[-1])
        elif self.__last_fe > self.__total_fes:
            raise ValueError(
                f"Last FE {self.__last_fe} inconsistent with total number"
                f"{self.__total_fes} of FEs.")

        ff: np.ndarray
        if self.__f_name == F_NAME_RAW:
            ff = np.array(self.__f_collector)
        elif self.__f_name == F_NAME_SCALED:
            ff = np.array([f / f_standard for f in self.__f_collector])
        else:
            ff = np.array([(f - f_standard) / f_standard
                           for f in self.__f_collector])
        self.__f_collector.clear()

        self.__consumer(Progress(self.algorithm,
                                 self.instance,
                                 self.rand_seed,
                                 np.array(self.__t_collector),
                                 self.__time_unit,
                                 ff,
                                 self.__f_name,
                                 f_standard,
                                 self.__only_improvements))

        self.__total_time = None
        self.__goal_f = None
        self.__t_collector.clear()
        self.__total_time = None
        self.__total_fes = None
        self.__last_fe = None
        self.__state = 0
        return super().end_file()

    def start_section(self, title: str) -> bool:
        super().start_section(title)
        if title == logging.SECTION_SETUP:
            if (self.__state & 1) != 0:
                raise ValueError(f"Already did section {title}.")
            self.__state |= 8
            return True
        if title == logging.SECTION_FINAL_STATE:
            if (self.__state & 2) != 0:
                raise ValueError(f"Already did section {title}.")
            self.__state |= 16
            return True
        if title == logging.SECTION_PROGRESS:
            if (self.__state & 4) != 0:
                raise ValueError(f"Already did section {title}.")
            self.__state |= 32
            return True
        return False

    def lines(self, lines: list[str]) -> bool:
        if not isinstance(lines, list):
            raise type_error(lines, "lines", list)

        if (self.__state & 24) != 0:  # final state or setup
            data = parse_key_values(lines)

            if not isinstance(data, dict):
                raise ValueError("Error when parsing data.")

            if (self.__state & 8) != 0:  # state
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

                seed_check = int(data[_FULL_KEY_RAND_SEED])
                if seed_check != self.rand_seed:
                    raise ValueError(
                        f"Found seed {seed_check} in log file, but file name "
                        f"indicates seed {self.rand_seed}.")

                self.__state = (self.__state | 1) & (~8)
                return self.__state != 7

            if (self.__state & 16) != 0:  # END_STATE
                self.__total_fes = int(data[logging.KEY_TOTAL_FES])
                self.__total_time = \
                    int(data[logging.KEY_TOTAL_TIME_MILLIS])
                self.__state = (self.__state | 2) & (~16)
                return self.__state != 7

        if (self.__state & 32) != 0:  # CSV data
            n_rows = len(lines)
            if n_rows < 2:
                raise ValueError("lines must contain at least two elements,"
                                 f"but contains {n_rows}.")

            columns = [c.strip() for c in
                       lines[0].split(CSV_SEPARATOR)]
            n_cols = len(columns)
            if n_cols < 3:
                raise ValueError("There must be at least three columns, "
                                 f"but found {n_cols} in '{lines[0]}'.")

            time_col_name: str = logging.PROGRESS_TIME_MILLIS if \
                self.__time_unit == TIME_UNIT_MILLIS else logging.PROGRESS_FES
            time_col_idx: int = -1
            f_col_idx: int = -1
            fe_col_idx: int = -1
            for idx, col in enumerate(columns):  # find the columns we
                if col == logging.PROGRESS_FES:
                    fe_col_idx = idx
                if col == time_col_name:
                    if time_col_idx >= 0:
                        raise ValueError(f"Time column {time_col_name} "
                                         "appears twice.")
                    time_col_idx = idx
                elif col == logging.PROGRESS_CURRENT_F:
                    if f_col_idx >= 0:
                        raise ValueError(
                            f"F column {logging.PROGRESS_CURRENT_F} "
                            "appears twice.")
                    f_col_idx = idx

            def aa(splt):  # noqa
                return splt[time_col_idx], splt[f_col_idx]

            time, f = zip(*[[c.strip()
                             for c in aa(line.split(CSV_SEPARATOR))]
                            for line in lines[1:]])
            time = [int(t) for t in time]
            f = [str_to_intfloat(v) for v in f]
            if self.__only_improvements:
                biggest_t: int = -1
                best_f: int | float = inf
                for idx, t in enumerate(time):
                    v = f[idx]
                    if t > biggest_t:
                        if biggest_t >= 0:
                            self.__t_collector.append(biggest_t)
                            self.__f_collector.append(best_f)
                        best_f = v
                        biggest_t = t
                    elif v < best_f:
                        best_f = v
                if biggest_t >= 0:
                    self.__t_collector.append(biggest_t)
                    self.__f_collector.append(best_f)
            else:
                self.__t_collector.extend(time)
                self.__f_collector.extend(f)

            self.__last_fe = int((lines[-1].split(CSV_SEPARATOR))[fe_col_idx])
            if self.__last_fe <= 0:
                raise ValueError(f"Last FE cannot be {self.__last_fe}.")

            self.__state = (self.__state | 4) & (~32)
            return self.__state != 7

        raise ValueError("Illegal state.")
