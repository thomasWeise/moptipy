"""
Objects embodying the progress of a run over time.

An instance of :class:`Progress` holds one :attr:`~Progress.time` vector and
an objective value (:attr:`~Progress.f`) vector. The time dimension (stored in
:attr:`~Progress.time_unit`) can either be in FEs or in milliseconds and the
objective value dimension (stored in :attr:`~Progress.f_name`) can be raw
objective values, standardized objective values, or normalized objective
values.
The two vectors together thus describe how a run of an optimization algorithm
improves the objective value over time.
"""
from dataclasses import dataclass
from math import inf, isfinite
from typing import Any, Callable, Final, Generator, Iterable

import numpy as np
from pycommons.io.console import logger
from pycommons.io.csv import COMMENT_START, CSV_SEPARATOR
from pycommons.io.path import Path
from pycommons.strings.string_conv import num_to_str, str_to_num
from pycommons.types import type_error

from moptipy.api.logging import (
    KEY_ALGORITHM,
    KEY_GOAL_F,
    KEY_INSTANCE,
    KEY_RAND_SEED,
    PROGRESS_CURRENT_F,
    PROGRESS_FES,
    PROGRESS_TIME_MILLIS,
    SECTION_PROGRESS,
)
from moptipy.evaluation.base import (
    F_NAME_RAW,
    F_NAME_SCALED,
    KEY_ENCODING,
    KEY_OBJECTIVE_FUNCTION,
    TIME_UNIT_FES,
    TIME_UNIT_MILLIS,
    PerRunData,
    check_f_name,
    check_time_unit,
)
from moptipy.evaluation.log_parser import SetupAndStateParser
from moptipy.utils.logger import (
    KEY_VALUE_SEPARATOR,
)
from moptipy.utils.nputils import is_all_finite, is_np_float, is_np_int


@dataclass(frozen=True, init=False, order=False, eq=False)
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
                 objective: str,
                 encoding: str | None,
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
        :param objective: the name of the objective function
        :param encoding: the name of the encoding that was used, if any, or
            `None` if no encoding was used
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
        super().__init__(algorithm, instance, objective, encoding, rand_seed)

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
            elif np.any(time[1:] < time[:-1]):
                raise ValueError("time data must be monotonously"
                                 f"increasing, but encountered {time}.")

        object.__setattr__(self, "time", time)
        object.__setattr__(self, "time_unit", check_time_unit(time_unit))

        mintime = 1 if time_unit == TIME_UNIT_FES else 0
        if np.any(time < mintime):
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


def to_csv(progress: Progress, file: str,
           put_header: bool = True) -> str:
    """
    Store a :class:`Progress` record in a CSV file.

    :param file: the file to generate
    :param put_header: should we put a header with meta-data?
    :return: the fully resolved file name
    """
    if not isinstance(progress, Progress):
        raise type_error(progress, "progress", Progress)
    if not isinstance(put_header, bool):
        raise type_error(put_header, "put_header", bool)
    path: Final[Path] = Path(file)
    logger(f"Writing progress object to CSV file {path!r}.")
    path.ensure_parent_dir_exists()

    with path.open_for_write() as out:
        sep: Final[str] = CSV_SEPARATOR
        write: Final[Callable[[str], int]] = out.write
        if put_header:
            kv: Final[str] = KEY_VALUE_SEPARATOR
            cmt: Final[str] = COMMENT_START
            write(f"{cmt} {KEY_ALGORITHM}{kv}{progress.algorithm}\n")
            write(f"{cmt} {KEY_INSTANCE}{kv}{progress.instance}\n")
            write(f"{cmt} {KEY_OBJECTIVE_FUNCTION}{kv}{progress.objective}\n")
            if progress.encoding is not None:
                write(f"{cmt} {KEY_ENCODING}{kv}{progress.objective}\n")
            write(f"{cmt} {KEY_RAND_SEED}{kv}{hex(progress.rand_seed)}\n")
            if progress.f_standard is not None:
                write(f"{cmt} {KEY_GOAL_F}{kv}{progress.f_standard}\n")
        write(f"{progress.time_unit}{sep}{progress.f_name}\n")
        for i, t in enumerate(progress.time):
            write(f"{t}{sep}{num_to_str(progress.f[i])}\n")

    logger(f"Done writing progress object to CSV file {path!r}.")

    path.enforce_file()
    return path


class __InnerLogParser(SetupAndStateParser[Progress]):
    """The internal log parser class."""

    def __init__(self, time_unit: str, f_name: str,
                 f_standard: dict[str, int | float] | None,
                 only_improvements: bool,
                 path_filter: Callable[[Path], bool] | None) -> None:
        """
        Create the internal log parser.

        :param time_unit: the time unit
        :param f_name: the objective name
        :param f_standard: a dictionary mapping instances to standard values
        :param only_improvements: enforce that f-values should be improving
            and time values increasing
        :param path_filter: the path filter
        """
        super().__init__(path_filter)
        self.__time_unit = check_time_unit(time_unit)
        self.__f_name = check_f_name(f_name)
        self.__last_fe: int | None = None
        self.__t_collector: Final[list[int]] = []
        self.__f_collector: Final[list[int | float]] = []
        if not isinstance(only_improvements, bool):
            raise type_error(only_improvements, "only_improvements", bool)
        self.__only_improvements = only_improvements
        if (f_standard is not None) and (not isinstance(f_standard, dict)):
            raise type_error(f_standard, "f_standard", dict)
        self.__f_standard: Final[None | dict[str, int | float]] \
            = f_standard
        self.__state: int = 0

    def _parse_file(self, file: Path) -> Progress | None:
        super()._parse_file(file)
        if self.__state != 2:
            raise ValueError(
                "Illegal state, log file must have a "
                f"{SECTION_PROGRESS!r} section.")
        if not self.__f_collector:
            raise ValueError("f-collector cannot be empty.")
        if not self.__t_collector:
            raise ValueError("time-collector cannot be empty.")
        self.__state = 0

        f_standard: int | float | None = None
        if (self.__f_standard is not None) and \
                (self.instance in self.__f_standard):
            f_standard = self.__f_standard[self.instance]
        if f_standard is None:
            f_standard = self.goal_f
        if (self.__f_name != F_NAME_RAW) and (f_standard is None):
            raise ValueError(f"f_standard cannot be {f_standard} if f_name "
                             f"is {self.__f_name}.")
        tt = self.total_time_millis if (self.__time_unit == TIME_UNIT_MILLIS) \
            else self.total_fes
        if tt < self.__t_collector[-1]:
            raise ValueError(
                f"Last time units {tt} inconsistent with last"
                f"recorded time unit {self.__t_collector[-1]}.")
        if self.__last_fe < self.total_fes:
            if tt > self.__t_collector[-1]:
                self.__t_collector.append(tt)
                self.__f_collector.append(self.__f_collector[-1])
        elif self.__last_fe > self.total_fes:
            raise ValueError(
                f"Last FE {self.__last_fe} inconsistent with total number"
                f"{self.total_fes} of FEs.")

        ff: np.ndarray
        if self.__f_name == F_NAME_RAW:
            ff = np.array(self.__f_collector)
        elif self.__f_name == F_NAME_SCALED:
            ff = np.array([f / f_standard for f in self.__f_collector])
        else:
            ff = np.array([(f - f_standard) / f_standard
                           for f in self.__f_collector])
        self.__f_collector.clear()

        return Progress(self.algorithm,
                        self.instance,
                        self.objective,
                        self.encoding,
                        self.rand_seed,
                        np.array(self.__t_collector),
                        self.__time_unit,
                        ff,
                        self.__f_name,
                        f_standard,
                        self.__only_improvements)

    def _end_parse_file(self, file: Path) -> None:
        """Clean up."""
        self.__t_collector.clear()
        self.__last_fe = None
        super()._end_parse_file(file)

    def _start_section(self, title: str) -> bool:
        if title == SECTION_PROGRESS:
            if self.__state != 0:
                raise ValueError(f"Already did section {title}.")
            self.__state = 1
            return True
        return super()._start_section(title)

    def _needs_more_lines(self) -> bool:
        return (self.__state < 2) or super()._needs_more_lines()

    def _lines(self, lines: list[str]) -> bool:
        if not isinstance(lines, list):
            raise type_error(lines, "lines", list)
        if self.__state != 1:
            return super()._lines(lines)
        n_rows = len(lines)
        if n_rows < 2:
            raise ValueError("lines must contain at least two elements,"
                             f"but contains {n_rows}.")

        columns = [c.strip() for c in lines[0].split(CSV_SEPARATOR)]
        n_cols = len(columns)
        if n_cols < 3:
            raise ValueError("There must be at least three columns, "
                             f"but found {n_cols} in {lines[0]!r}.")

        time_col_name: str = PROGRESS_TIME_MILLIS if \
            self.__time_unit == TIME_UNIT_MILLIS else PROGRESS_FES
        time_col_idx: int = -1
        f_col_idx: int = -1
        fe_col_idx: int = -1
        for idx, col in enumerate(columns):  # find the columns we
            if col == PROGRESS_FES:
                fe_col_idx = idx
            if col == time_col_name:
                if time_col_idx >= 0:
                    raise ValueError(f"Time column {time_col_name} "
                                     "appears twice.")
                time_col_idx = idx
            elif col == PROGRESS_CURRENT_F:
                if f_col_idx >= 0:
                    raise ValueError(
                        f"F column {PROGRESS_CURRENT_F} "
                        "appears twice.")
                f_col_idx = idx

        def aa(splt):  # noqa
            return splt[time_col_idx], splt[f_col_idx]

        time: Iterable[int]
        f: Iterable[Any]
        time, f = zip(*[[c.strip()
                         for c in aa(line.split(CSV_SEPARATOR))]
                        for line in lines[1:]], strict=True)
        time = [int(t) for t in time]
        f = [str_to_num(v) for v in f]
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

        self.__state = 2
        return self._needs_more_lines()


def from_logs(path: str,
              time_unit: str = TIME_UNIT_FES,
              f_name: str = F_NAME_RAW,
              f_standard: dict[str, int | float] | None = None,
              only_improvements: bool = True,
              path_filter: Callable[[Path], bool] | None = None) \
        -> Generator[Progress, None, None]:
    """
    Parse a given path and pass yield all progress data found.

    If `path` identifies a file with suffix `.txt`, then this file is
    parsed. The appropriate :class:`Progress` is created. If `path` identifies
    a directory, then this directory is parsed recursively for each log file
    found, one record is returned.

    :param path: the path to parse
    :param time_unit: the time unit
    :param f_name: the objective name
    :param f_standard: a dictionary mapping instances to standard values
    :param only_improvements: enforce that f-values should be improving and
        time values increasing
    :param path_filter: a function to filter paths
    """
    return __InnerLogParser(time_unit, f_name, f_standard,
                            only_improvements, path_filter).parse(path)
