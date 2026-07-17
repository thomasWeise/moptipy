"""A set of end results from a multi-objective run."""

import argparse
from dataclasses import dataclass
from itertools import chain
from typing import Any, Callable, Final, Generator, Iterable, cast

from pycommons.ds.sequences import reiterable
from pycommons.io.console import logger
from pycommons.io.csv import (
    CSV_SEPARATOR,
    csv_column,
    csv_scope,
    csv_val_or_none,
)
from pycommons.io.parser import Parser
from pycommons.io.path import Path, file_path, write_lines
from pycommons.strings.string_conv import (
    num_to_str,
    str_to_num,
)
from pycommons.types import type_error

from moptipy.api.logging import (
    PREFIX_SECTION_ARCHIVE,
    SECTION_ARCHIVE_QUALITY,
    SECTION_PROGRESS,
    SUFFIX_SECTION_ARCHIVE_X,
    SUFFIX_SECTION_ARCHIVE_Y,
)
from moptipy.evaluation.end_results import CsvReader as CsvReaderBase
from moptipy.evaluation.end_results import CsvWriter as CsvWriterBase
from moptipy.evaluation.end_results import EndResult
from moptipy.evaluation.end_results import EndResultLogParser as _Erlp
from moptipy.utils.help import moptipy_argparser
from moptipy.utils.logger import (
    SECTION_END,
    SECTION_START,
)
from moptipy.utils.math import try_int


@dataclass(frozen=True, init=False, order=False, eq=False)
class MOEndResult(EndResult):
    """A multi-objective end result record."""

    #: The objective values for the subordinate objective functions.
    fs: tuple[int | float, ...]

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
                 max_time_millis: int | None,
                 fs: tuple[int | float, ...],
                 x: str | None = None,
                 y: str | None = None) -> None:
        """
        Create the multi-objective end result record.

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
        :param fs: the objective value vector
        :param x: the optional point in the search space
        :param y: the optional point in the solution space

        :raises TypeError: if any parameter has a wrong type
        :raises ValueError: if the parameter values are inconsistent
        """
        super().__init__(
            algorithm=algorithm,
            instance=instance,
            objective=objective,
            encoding=encoding,
            rand_seed=rand_seed,
            best_f=best_f,
            last_improvement_fe=last_improvement_fe,
            last_improvement_time_millis=last_improvement_time_millis,
            total_fes=total_fes,
            total_time_millis=total_time_millis,
            goal_f=goal_f,
            max_fes=max_fes,
            max_time_millis=max_time_millis,
            x=x,
            y=y)
        fsc = tuple.__len__(fs)
        if fsc <= 0:
            raise ValueError("Number of objectives must be greater than 0.")
        fsu: list[int | float] = []
        changed: bool = False
        for val in fs:
            val2 = try_int(val)
            if val2 is not val:
                changed = True
            fsu.append(val2)
        object.__setattr__(self, "fs", tuple(fsu) if changed else fs)

    def _tuple(self) -> tuple[Any, ...]:
        """
        Get the comparison tuple.

        :return: the comparison tuple
        """
        cr = list(super()._tuple())
        cr.extend(self.fs)
        return tuple(cr)


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
        write_lines(CsvWriter.write(results), wt)
    logger(f"Done writing end results to CSV file {path!r}.")
    return path


def from_csv(file: str,
             filterer: Callable[[EndResult], bool]
             = lambda _: True) -> Generator[
        EndResult | MOEndResult, None, None]:
    """
    Parse a given CSV file to get :class:`MOEndResult` Records.

    :param file: the path to parse
    :param filterer: an optional filter function
    """
    path: Final[Path] = file_path(file)
    logger(f"Now reading CSV file {path!r}.")
    with path.open_for_read() as rd:
        for r in CsvReader.read(rd):
            if filterer(r):
                yield r
    logger(f"Done reading CSV file {path!r}.")


def from_logs(path: str, parse_x: bool = True, parse_y: bool = True) \
        -> Generator[EndResult | MOEndResult, None, None]:
    """
    Parse a given path and yield all (multi-objective) end results found.

    If `path` identifies a file with suffix `.txt`, then this file is
    parsed. The appropriate :class:`moptipy.evaluation.end_results.EndResult`
    or :class:`MOEndResult` is created and yielded.
    If `path` identifies a directory, then this directory is parsed
    recursively for each log file found, one record is yielded.

    :param path: the path to parse
    :param parse_x: should we parse the points in the search space, too?
    :param parse_y: should we parse the points in the solution space, too?
    """
    for group in __MOEndResultLogParser(parse_x, parse_y).parse(path):
        yield from group


class CsvWriter(CsvWriterBase):
    """A class for CSV writing of `EndResult` records."""

    def __init__(self, data: Iterable[EndResult],
                 scope: str | None = None) -> None:
        """
        Initialize the csv writer.

        :param data: the data
        :param scope: the prefix to be pre-pended to all columns
        """
        data = reiterable(data)
        super().__init__(data, scope)

        #: do we need the encoding?
        self.__fcols: Final[int] = max(
            tuple.__len__(er.fs) if isinstance(
                er, MOEndResult) else 0 for er in data)

    def get_column_titles(self) -> Iterable[str]:
        """
        Get the column titles.

        :returns: the column titles
        """
        p: Final[str | None] = self.scope
        return chain(super().get_column_titles(), (
            csv_scope(p, x) for x in (
                f"f{i}" for i in range(self.__fcols))))

    def get_row(self, data: EndResult) -> Iterable[str]:
        """
        Render a single end result record to a CSV row.

        :param data: the end result record
        :returns: the row iterator
        """
        yield from super().get_row(data)
        if isinstance(data, MOEndResult):
            yield from map(num_to_str, data.fs)

    def get_header_comments(self) -> Iterable[str]:
        """
        Get any possible header comments.

        :returns: the header comments
        """
        return ("Multi-Objective Experiment End Results",
                "See the description at the bottom of the file.")

    def get_footer_comments(self) -> Iterable[str]:
        """
        Get any possible footer comments.

        :returns: the footer comments
        """
        yield from super().get_footer_comments()
        for i in range(self.__fcols):
            yield (f"f{i}: the objective value computed with "
                   f"the {i + 1}-th objective function.")


class CsvReader(CsvReaderBase):
    """A csv parser for end results."""

    def __init__(self, columns: dict[str, int]) -> None:
        """
        Create a CSV parser for `EndResult` records.

        :param columns: the columns
        """
        super().__init__(columns)
        i: int = 0
        fcols: Final[list[int]] = []
        while True:
            colname: str = f"f{i}"
            try:
                fcols.append(csv_column(columns, colname))
            except KeyError:
                break
            i += 1
        #: the objective value columns
        self.__fcols: Final[tuple[int, ...]] = tuple(fcols)

    def parse_row(self, data: list[str]) -> EndResult | MOEndResult:
        """
        Parse a row of data.

        :param data: the data row
        :return: the end result statistics
        """
        res = super().parse_row(data)
        vals: Final[list[int | float]] = []
        for col in self.__fcols:
            v = csv_val_or_none(data, col, str_to_num)
            if v is None:
                break
            vals.append(v)
        if list.__len__(vals) <= 0:
            return res
        return MOEndResult(
            algorithm=res.algorithm,
            instance=res.instance,
            objective=res.objective,
            encoding=res.encoding,
            rand_seed=res.rand_seed,
            best_f=res.best_f,
            last_improvement_fe=res.last_improvement_fe,
            last_improvement_time_millis=res.last_improvement_time_millis,
            total_fes=res.total_fes,
            total_time_millis=res.total_time_millis,
            goal_f=res.goal_f,
            max_fes=res.max_fes,
            max_time_millis=res.max_time_millis,
            fs=tuple(vals),
            x=res.x,
            y=res.y)


class __MOEndResultLogParser(Parser[Iterable[EndResult]]):
    """The internal log parser class."""

    def __init__(self, parse_x: bool = True, parse_y: bool = True) -> None:
        """
        Parse the log files.

        :param parse_x: whether to parse the x values
        :param parse_y: whether to parse the y values
        """
        super().__init__()
        if not isinstance(parse_x, bool):
            raise type_error(parse_x, "parse_x", bool)
        if not isinstance(parse_y, bool):
            raise type_error(parse_y, "parse_y", bool)
        #: shall we parse the points in the search space?
        self.__parse_x: Final[bool] = parse_x
        #: shall we parse the points in the solution space?
        self.__parse_y: Final[bool] = parse_y

    def _parse_file(self, file: Path) -> Iterable[EndResult]:
        """
        Get the parsing result.

        :returns: the `EndResult` instance
        """
        self._progress_logger(
            f"Beginning multi-objective parsing of file {file!r}.")
        o: Final[EndResult] = _Erlp().parse_file(file)
        if not isinstance(o, EndResult):
            raise type_error(o, f"parse({file!r})", EndResult)

        with file.open_for_read() as reader:
            lines: tuple[str, ...] = tuple(map(str.strip, str.splitlines(
                reader.read())))
        count = tuple.__len__(lines)
        if count <= 2:
            raise ValueError(
                f"Inconsistent number {count} of lines in file {file!r}")

        # first, we process the archive to find all the retained points
        archive: Final[list[tuple[int | float, ...]]] = []
        begin: str = f"{SECTION_START}{SECTION_ARCHIVE_QUALITY}"
        end: str = f"{SECTION_END}{SECTION_ARCHIVE_QUALITY}"
        state: int = 0
        for line in lines:
            if line == begin:
                if state != 0:
                    raise ValueError(f"Inconsistent begin state in "
                                     f"file {file!r} vs. {begin!r}/{end!r}.")
                state = 1
                continue
            if line == end:
                if state != 2:
                    raise ValueError(f"Inconsistent end state in "
                                     f"file {file!r} vs. {begin!r}/{end!r}.")
                state = 3
                break
            if state == 1:
                if line.startswith("f"):
                    state = 2
                    continue
                state = 2
            if state != 2:
                continue
            try:
                archive.append(tuple(map(str_to_num, map(str.strip, str.split(
                    line, CSV_SEPARATOR)))))
            except ValueError as ve:
                raise ValueError(
                    f"Error when parsing line {line!r} of file {file!r} in "
                    f"{SECTION_ARCHIVE_QUALITY}.") from ve
        if state != 3:
            return (o, )

        # now, we find the progress to find out when the solutions emerged
        progress: Final[list[tuple[int | float, ...]]] = []
        begin = f"{SECTION_START}{SECTION_PROGRESS}"
        end = f"{SECTION_END}{SECTION_PROGRESS}"
        state = 0
        for line in lines:
            if line == begin:
                if state != 0:
                    raise ValueError(f"Inconsistent begin state in "
                                     f"file {file!r} vs. {begin!r}/{end!r}.")
                state = 1
                continue
            if line == end:
                if state != 2:
                    raise ValueError(f"Inconsistent end state in "
                                     f"file {file!r} vs. {begin!r}/{end!r}.")
                state = 3
                break
            if state == 1:
                if line.startswith("fes"):
                    state = 2
                    continue
                state = 2
            if state != 2:
                continue
            try:
                progress.append(tuple(map(str_to_num, map(str.strip, str.split(
                    line, CSV_SEPARATOR)))))
            except ValueError as ve:
                raise ValueError(
                    f"Error when parsing line {line!r} of file {file!r} "
                    f"in {SECTION_PROGRESS}.") from ve
        if state not in {0, 3}:
            raise ValueError(f"Inconsistent state {state} in "
                             f"file {file!r} vs. {begin!r}/{end!r}.")

        count = list.__len__(archive)
        if count < 1:
            raise ValueError(f"No solution archived in file {file!r}.")

        # Now we try to find the solutions and points matching them
        x: dict[int, str] = {}
        y: dict[int, str] = {}
        if self.__parse_x or self.__parse_y:
            current: list[str] = []
            current_id: int = 0
            state = 0
            arc_start: Final[str] = f"{SECTION_START}{PREFIX_SECTION_ARCHIVE}"
            arc_end: Final[str] = f"{SECTION_END}{PREFIX_SECTION_ARCHIVE}"

            for line in lines:
                if str.startswith(line, arc_start):
                    if state != 0:
                        raise ValueError(
                            f"Inconsistent start state in {file!r}, "
                            f"encountered {line!r}.")
                    if str.endswith(line, SUFFIX_SECTION_ARCHIVE_X):
                        current_id = int(line[len(arc_start):-len(
                            SUFFIX_SECTION_ARCHIVE_X)])
                        if current_id in x:
                            raise ValueError(
                                f"Encountered archive X {current_id} twice "
                                f"in {file!r}.")
                        state = 1
                        current.clear()
                        continue
                    if str.endswith(line, SUFFIX_SECTION_ARCHIVE_Y):
                        current_id = int(line[len(arc_start):-len(
                            SUFFIX_SECTION_ARCHIVE_Y)])
                        if current_id in x:
                            raise ValueError(
                                f"Encountered archive Y {current_id} twice "
                                f"in {file!r}.")
                        state = 2
                        current.clear()
                        continue
                if state in {1, 2}:
                    if str.startswith(line, arc_end):
                        if state == 1:
                            if self.__parse_x:
                                x[current_id] = "\n".join(current)
                        elif (state == 2) and self.__parse_y:
                            y[current_id] = "\n".join(current)
                        current.clear()
                        state = 0
                        continue
                    current.append(line)

            if state != 0:
                raise ValueError(f"Inconsistent state in file {file!r}.")

        dim: Final[int] = tuple.__len__(archive[0])
        for solution in archive:
            if tuple.__len__(solution) != dim:
                raise ValueError(
                    f"Inconsistent archive dimension of {solution} in file "
                    f"{file!r}, should be {dim}.")

        for time in progress:
            if tuple.__len__(time) != dim + 2:
                raise ValueError(
                    "Inconsistent progress dimension of record "
                    f"{time} in {file!r}, should be {dim + 2}.")

        out: list[MOEndResult] = []
        for i, solution in enumerate(archive):
            found: tuple[int | float, ...] | None = None
            for time in progress:
                if time[-dim:] == solution:
                    found = time
                    break
            out.append(MOEndResult(
                algorithm=o.algorithm,
                instance=o.instance,
                objective=o.objective,
                encoding=o.encoding,
                rand_seed=o.rand_seed,
                best_f=solution[0],
                last_improvement_fe=o.last_improvement_fe
                if found is None else cast("int", found[0]),
                last_improvement_time_millis=o.last_improvement_time_millis
                if found is None else cast("int", found[1]),
                total_fes=o.total_fes,
                total_time_millis=o.total_time_millis,
                goal_f=o.goal_f,
                max_fes=o.max_fes,
                max_time_millis=o.max_time_millis,
                fs=solution[1:],
                x=x.get(i),
                y=y.get(i)))
        self._progress_logger(f"Done parsing file {file!r} multi-objectively.")
        return out


# Run log files to end results if executed as script
if __name__ == "__main__":
    parser: Final[argparse.ArgumentParser] = moptipy_argparser(
        __file__,
        "Convert multi-objective log files obtained with moptipy to the "
        "end results CSV format that can be post-processed or exported to "
        "other tools.",
        "This program recursively parses a folder hierarchy created by"
        " the moptipy multi-objective experiment execution facility. "
        "This folder structure follows the scheme of algorithm/instance/"
        "log_file and has one log file per run. As result of the parsing, "
        "one CSV file (where columns are separated by ';') is created with"
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
    args: Final[argparse.Namespace] = parser.parse_args()

    to_csv(from_logs(args.source), args.dest)
