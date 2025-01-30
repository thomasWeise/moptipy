"""
Parsers for structured log data produced by the `moptipy` experiment API.

The `moptipy` :class:`~moptipy.api.execution.Execution` and experiment-running
facility (:func:`~moptipy.api.experiment.run_experiment`) uses the class
:class:`~moptipy.utils.logger.Logger` from module :mod:`~moptipy.utils.logger`
to produce log files complying with
https://thomasweise.github.io/moptipy/#log-files.

Here we provide a skeleton for parsing such log files in form of the class
:class:`~LogParser`. It works similar to SAX-XML parsing in that the data
is read is from files and methods that consume the data are invoked. By
overwriting these methods, we can do useful things with the data.

For example in module :mod:`~moptipy.evaluation.end_results`, the method
:meth:`~moptipy.evaluation.end_results.from_logs` can load
:class:`~moptipy.evaluation.end_results.EndResult` records from the logs
and the method :meth:`~moptipy.evaluation.progress.from_logs` in
module :mod:`~moptipy.evaluation.progress` reads the whole
:class:`~moptipy.evaluation.progress.Progress` that the algorithms make
over time.
"""

from contextlib import suppress
from math import inf, isfinite, isinf
from typing import Callable, Final, TypeVar

from pycommons.io.csv import COMMENT_START, SCOPE_SEPARATOR
from pycommons.io.parser import Parser
from pycommons.io.path import Path
from pycommons.strings.string_conv import str_to_num
from pycommons.types import check_to_int_range, type_error

from moptipy.api.logging import (
    ERROR_SECTION_PREFIX,
    FILE_SUFFIX,
    KEY_BEST_F,
    KEY_GOAL_F,
    KEY_LAST_IMPROVEMENT_FE,
    KEY_LAST_IMPROVEMENT_TIME_MILLIS,
    KEY_MAX_FES,
    KEY_MAX_TIME_MILLIS,
    KEY_NAME,
    KEY_RAND_SEED,
    KEY_TOTAL_FES,
    KEY_TOTAL_TIME_MILLIS,
    SCOPE_ALGORITHM,
    SCOPE_ENCODING,
    SCOPE_OBJECTIVE_FUNCTION,
    SCOPE_PROCESS,
    SECTION_FINAL_STATE,
    SECTION_SETUP,
)
from moptipy.evaluation._utils import _check_max_time_millis
from moptipy.utils.logger import (
    SECTION_END,
    SECTION_START,
    parse_key_values,
)
from moptipy.utils.nputils import rand_seed_check
from moptipy.utils.strings import (
    PART_SEPARATOR,
    sanitize_name,
)

#: the maximum FEs of a black-box process
_FULL_KEY_MAX_FES: Final[str] = \
    f"{SCOPE_PROCESS}{SCOPE_SEPARATOR}{KEY_MAX_FES}"
#: the maximum runtime in milliseconds of a black-box process
_FULL_KEY_MAX_TIME_MILLIS: Final[str] = \
    f"{SCOPE_PROCESS}{SCOPE_SEPARATOR}{KEY_MAX_TIME_MILLIS}"
#: the goal objective value of a black-box process
_FULL_KEY_GOAL_F: Final[str] = f"{SCOPE_PROCESS}{SCOPE_SEPARATOR}{KEY_GOAL_F}"
#: the random seed
_FULL_KEY_RAND_SEED: Final[str] = \
    f"{SCOPE_PROCESS}{SCOPE_SEPARATOR}{KEY_RAND_SEED}"
#: the full algorithm name key
_FULL_KEY_ALGORITHM: Final[str] = \
    f"{SCOPE_ALGORITHM}{SCOPE_SEPARATOR}{KEY_NAME}"
#: the full objective function name key
_FULL_KEY_OBJECTIVE: Final[str] = \
    f"{SCOPE_OBJECTIVE_FUNCTION}{SCOPE_SEPARATOR}{KEY_NAME}"
#: the full encoding name key
_FULL_KEY_ENCODING: Final[str] = \
    f"{SCOPE_ENCODING}{SCOPE_SEPARATOR}{KEY_NAME}"


def _true(_) -> bool:
    """
    Get `True` as return value, always.

    :retval `True`: always
    """
    return True


#: the type variable for data to be read from the directories
T = TypeVar("T")


class LogParser[T](Parser[T]):
    """
    A log parser can parse a log file and separate the sections.

    The log parser is designed to load data from text files generated
    by :class:`~moptipy.utils.logger.FileLogger`. It can also recursively
    parse directories.
    """

    def __init__(self, path_filter: Callable[[Path], bool] | None = None):
        """
        Initialize the log parser.

        :param path_filter: a filter allowing us to skip paths or files. If
            this :class:`Callable` returns `True`, the file or directory is
            considered for parsing. If it returns `False`, it is skipped.
        """
        if path_filter is None:
            path_filter = _true
        elif not callable(path_filter):
            raise type_error(path_filter, "path_filter", call=True)
        #: the current depth in terms of directories
        self.__depth: int = 0
        #: the path filter
        self.__path_filter: Final[Callable[[Path], bool]] = path_filter

    def start_list_dir(self, root: Path, current: Path) -> tuple[bool, bool]:
        """
        Decide whether to enter a directory to parse all files inside.

        :param root: the root directory
        :param current: the path of the directory
        :return: a tuple with two `True` values if all the sub-directories and
            files inside the directory should be processed, two `False` values
            if this directory should be skipped and parsing should continue
            with the next sibling directory
        """
        should: Final[bool] = self.__path_filter(current)
        return should, should

    def start_parse_file(self, root: Path, current: Path) -> bool:
        """
        Decide whether to start parsing a file.

        :param root: the root path
        :param current: the file path
        :return: `True` if the file should be parsed, `False` if it should be
            skipped (and
            :meth:`~moptipy.evaluation.log_parser.LogParser.parse_file` should
            return `True`).
        """
        return current.endswith(FILE_SUFFIX) and self.__path_filter(current)

    # noinspection PyMethodMayBeStatic
    def start_section(self, title: str) -> bool:
        """
        Start a section.

        If this method returns `True`, then all the lines of text of the
        section `title` will be read and together passed to
        :meth:`~moptipy.evaluation.log_parser.LogParser.lines`.
        If this method returns `False`, then the section will be skipped
        and we fast-forward to the next section, if any.

        :param title: the section title
        :return: `True` if the section data should be loaded and passed to
            :meth:`lines`, `False` of the section can be skipped. In that
            case, we will fast-forward to the next
            :meth:`~moptipy.evaluation.log_parser.LogParser.start_section`.
        """
        if not title:
            raise ValueError(f"Title cannot be empty, but is {title!r}.")
        if title.startswith(ERROR_SECTION_PREFIX):
            raise ValueError(f"Encountered error section {title!r}.")
        return False

    # noinspection PyUnusedLocal
    # noinspection PyMethodMayBeStatic
    def lines(self, lines: list[str]) -> bool:
        """
        Consume all the lines from a section.

        This method receives the complete text of a section, where all lines
        are separated and put into one list `lines`. Each line is stripped
        from whitespace and comments, empty lines are omitted.
        If this method returns `True`, we will continue parsing the file and
        move to the next section, if any, or directly to the end of the file
        parsing process.

        :param lines: the lines to consume
        :return: `True` if further parsing is necessary and the next section
            should be fed to
            :meth:`~moptipy.evaluation.log_parser.LogParser.start_section`,
            `False` if the parsing process can be terminated`
        """
        del lines
        return True

    def before_get_result(self) -> None:
        """Initialize the result construction: *before* `get_result`."""

    def after_get_result(self) -> None:
        """Cleanup after result construction: *after* `get_result`."""

    def get_result(self) -> T | None:
        """
        Get the result of the file parsing process.

        :returns: the result of the file parsin process, or `None`
        """
        return None

    def parse_file(self, root: Path, current: Path) -> T | None:
        """
        Parse the contents of a file.

        :param root: the root path
        :param current: the file to parse
        :return: the return value received from invoking `get_result`
        """
        lines: list[str] = []
        buffer: list[str] = []
        state: int = 0
        wants_section: bool = False
        sec_end: str = ""
        section: str = ""
        sect_start: Final[str] = SECTION_START
        sect_end: Final[str] = SECTION_END
        cmt_chr: Final[str] = COMMENT_START

        index: int = 0
        with (current.open_for_read() as handle):
            while True:

                # get the next line
                if index >= len(buffer):
                    try:
                        buffer = handle.readlines(128)
                    except Exception as be:
                        raise ValueError(
                            f"Error when reading lines from file {current!r} "
                            f"while in section {section!r}."
                            if state == 1 else
                            "Error when reading lines from file "
                            f"{current!r}.") from be
                    if (buffer is None) or (len(buffer) <= 0):
                        break
                    index = 0

                orig_cur = buffer[index]
                index += 1

                # strip next line from comments and white space
                cur = orig_cur.strip()
                if len(cur) <= 0:
                    continue

                i = cur.find(cmt_chr)
                if i >= 0:
                    cur = cur[:i].strip()
                    if len(cur) <= 0:
                        continue

                if state in (0, 2):
                    if not cur.startswith(sect_start):
                        raise ValueError("Line should start with "
                                         f"{sect_start!r} but is "
                                         f"{orig_cur!r} in file {current!r}.")
                    section = cur[len(sect_start):]
                    if len(section) <= 0:
                        raise ValueError(
                            "Section title cannot be empty in "
                            f"{current!r}, but encountered {orig_cur!r}.")
                    state = 1
                    sec_end = sect_end + section
                    wants_section = self.start_section(section)
                elif state == 1:
                    if cur == sec_end:
                        state = 2
                        if wants_section:
                            try:
                                do_next = self.lines(lines)
                            except Exception as be:
                                raise ValueError(
                                    "Error when processing section "
                                    f"{section!r} in file {current!r}.") \
                                    from be
                            lines.clear()
                            if not do_next:
                                break
                    elif wants_section:
                        lines.append(cur)

        if state == 0:
            raise ValueError(f"Log file {current!r} contains no section.")
        if state == 1:
            raise ValueError(f"Log file {current!r} ended before"
                             f"encountering {sec_end!r}.")
        try:
            self.before_get_result()
            return self.get_result()
        finally:
            self.after_get_result()


class ExperimentParser[T](LogParser[T]):
    """A log parser following our pre-defined experiment structure."""

    def __init__(self, path_filter: Callable[[Path], bool] | None = None):
        """
        Initialize the experiment parser.

        :param path_filter: a filter allowing us to skip paths or files. If
            this :class:`Callable` returns `True`, the file or directory is
            considered for parsing. If it returns `False`, it is skipped.
        """
        super().__init__(path_filter=path_filter)

        #: The name of the algorithm to which the current log file belongs.
        self.algorithm: str | None = None
        #: The name of the instance to which the current log file belongs.
        self.instance: str | None = None
        #: The random seed of the current log file.
        self.rand_seed: int | None = None

    def start_parse_file(self, root: Path, current: Path) -> bool:
        """
        Decide whether to start parsing a file and setup meta-data.

        :param root: the root path
        :param current: the file path
        :return: `True` if the file should be parsed, `False` if it should be
            skipped (and
            :meth:`~moptipy.evaluation.log_parser.LogParser.parse_file` should
            return `True`).
        """
        if not super().start_parse_file(root, current):
            return False

        self.algorithm = None
        self.instance = None
        self.rand_seed = None
        stop: bool = True
        with suppress(Exception):
            inst_dir: Final[Path] = current.up()
            algo_dir: Final[Path] = inst_dir.up()
            stop = not (algo_dir.contains(inst_dir)
                        and inst_dir.contains(current))
        if stop:
            return False
        self.instance = sanitize_name(inst_dir.basename())
        self.algorithm = sanitize_name(algo_dir.basename())
        start = (f"{self.algorithm}{PART_SEPARATOR}"
                 f"{self.instance}{PART_SEPARATOR}0x")
        base = current.basename()
        if (not base.startswith(start)) and \
                (not base.casefold().startswith(start.casefold())):
            # case-insensitive comparison needed because of Windows
            raise ValueError(
                f"File name of {current!r} should start with {start!r}.")
        self.rand_seed = rand_seed_check(int(
            base[len(start):(-len(FILE_SUFFIX))], base=16))
        return True

    def after_get_result(self) -> None:
        """Finalize parsing a file."""
        self.rand_seed = None
        self.algorithm = None
        self.instance = None
        return super().after_get_result()


class SetupAndStateParser[T](ExperimentParser[T]):
    """
    A log parser which loads and processes the basic data from the logs.

    This parser processes the `SETUP` and `STATE` sections of a log file and
    stores the performance-related information in member variables.
    """

    def __init__(self, path_filter: Callable[[Path], bool] | None = None):
        """
        Create the basic data parser.

        :param path_filter: a filter allowing us to skip paths or files. If
            this :class:`Callable` returns `True`, the file or directory is
            considered for parsing. If it returns `False`, it is skipped.
        """
        super().__init__(path_filter)
        #: the total consumed runtime, in objective function evaluations
        self.total_fes: int | None = None
        #: the total consumed runtime in milliseconds
        self.total_time_millis: int | None = None
        #: the best objective function value encountered
        self.best_f: int | float | None = None
        #: the objective function evaluation when the last improvement
        #: happened, in milliseconds
        self.last_improvement_fe: int | None = None
        #: the time step when the last improvement happened, in milliseconds
        self.last_improvement_time_millis: int | None = None
        #: the goal objective value, if any
        self.goal_f: int | float | None = None
        #: the maximum permitted number of objective function evaluations,
        #: if any
        self.max_fes: int | None = None
        #: the maximum runtime limit in milliseconds, if any
        self.max_time_millis: int | None = None
        #: The name of the objective to which the current log file belongs.
        self.objective: str | None = None
        #: The name of the encoding to which the current log file belongs.
        self.encoding: str | None = None
        #: the internal state, an OR mask: 1=after setup section, 2=after
        #: state section, 4=in setup section, 8=in state section
        self.__state: int = 0

    def start_parse_file(self, root: Path, current: Path) -> bool:
        """
        Begin parsing the file identified by `path`.

        :param root: the root path
        :param current: the path identifying the file
        """
        if not super().start_parse_file(root, current):
            return False
        if self.__state != 0:
            raise ValueError(f"Illegal state when trying to parse {current}.")
        return True

    def before_get_result(self) -> None:
        """Check the state before `get_result` is called."""
        # perform sanity checks
        if self.__state != 3:
            raise ValueError(
                "Illegal state, log file must have both a "
                f"{SECTION_FINAL_STATE!r} and a "
                f"{SECTION_SETUP!r} section.")
        if self.rand_seed is None:
            raise ValueError("rand_seed is missing.")
        if self.algorithm is None:
            raise ValueError("algorithm is missing.")
        if self.instance is None:
            raise ValueError("instance is missing.")
        if self.objective is None:
            raise ValueError("objective is missing.")
        if self.total_fes is None:
            raise ValueError("total_fes is missing.")
        if self.total_time_millis is None:
            raise ValueError("total_time_millis is missing.")
        if self.best_f is None:
            raise ValueError("best_f is missing.")
        if self.last_improvement_fe is None:
            raise ValueError("last_improvement_fe is missing.")
        if self.last_improvement_time_millis is None:
            raise ValueError("last_improvement_time_millis is missing.")

    def after_get_result(self) -> None:
        """Finalize the state *after* `get_result`."""
        # now clear the data
        self.total_fes = None
        self.total_time_millis = None
        self.best_f = None
        self.last_improvement_fe = None
        self.last_improvement_time_millis = None
        self.goal_f = None
        self.max_fes = None
        self.max_time_millis = None
        self.objective = None
        self.encoding = None
        self.__state = 0
        return super().after_get_result()

    def needs_more_lines(self) -> bool:
        """
        Check whether we need to process more lines.

        You can overwrite this method if your parser parses additional log
        sections. Your overwritten method should return `True` if more
        sections except `STATE` and `SETUP` still need to be parsed and return
        `super().needs_more_lines()` otherwise.

        :returns: `True` if more data needs to be processed, `False` otherwise
        """
        return self.__state != 3

    def lines(self, lines: list[str]) -> bool:
        """
        Process the lines loaded from a section.

        If you process more sections, you should override this method. Your
        overridden method then can parse the data if you are in the right
        section. It should end with `return super().lines(lines)`.

        :param lines: the lines that have been loaded
        :returns: `True` if parsing should be continued, `False` otherwise
        """
        if (self.__state & 4) != 0:
            self.setup_section(parse_key_values(lines))
        elif (self.__state & 8) != 0:
            self.state_section(lines)
        return self.needs_more_lines()

    def start_section(self, title: str) -> bool:
        """
        Begin a section.

        :param title: the section title
        :returns: `True` if the text of the section should be processed,
            `False` otherwise
        """
        super().start_section(title)
        if title == SECTION_SETUP:
            if (self.__state & 1) != 0:
                raise ValueError(f"Already did section {title!r}.")
            self.__state |= 4
            return True
        if title == SECTION_FINAL_STATE:
            if (self.__state & 2) != 0:
                raise ValueError(f"Already did section {title}.")
            self.__state |= 8
            return True
        return False

    def setup_section(self, data: dict[str, str]) -> None:
        """
        Parse the data from the `setup` section.

        :param data: the parsed data
        """
        self.goal_f = None
        if _FULL_KEY_GOAL_F in data:
            goal_f = data[_FULL_KEY_GOAL_F]
            g: Final[int | float] = str_to_num(goal_f)
            if isfinite(g):
                self.goal_f = g
            elif not (isinf(g) and (g >= inf)):
                raise ValueError(
                    f"invalid goal f {goal_f}, which renders to {g}")

        if _FULL_KEY_MAX_FES in data:
            self.max_fes = check_to_int_range(
                data[_FULL_KEY_MAX_FES], _FULL_KEY_MAX_FES, 1,
                1_000_000_000_000_000)
        if _FULL_KEY_MAX_TIME_MILLIS in data:
            self.max_time_millis = check_to_int_range(
                data[_FULL_KEY_MAX_TIME_MILLIS], _FULL_KEY_MAX_TIME_MILLIS, 1,
                1_000_000_000_000)
        if _FULL_KEY_ALGORITHM in data:
            a = data[_FULL_KEY_ALGORITHM]
            if a != self.algorithm:
                # this error may occur under windows due to case-insensitive
                # file names
                if a.casefold() == self.algorithm.casefold():
                    self.algorithm = a  # rely on name from log file
                else:  # ok, case was not the issue - raise error
                    raise ValueError(
                        f"algorithm name from file name is {self.algorithm!r}"
                        f", but key {_FULL_KEY_ALGORITHM!r} gives {a!r}.")
        else:
            raise ValueError(f"key {_FULL_KEY_ALGORITHM!r} missing in file!")

        if _FULL_KEY_OBJECTIVE in data:
            self.objective = data[_FULL_KEY_OBJECTIVE]
        else:
            raise ValueError(f"key {_FULL_KEY_OBJECTIVE!r} missing in file!")

        self.encoding = data.get(_FULL_KEY_ENCODING)

        seed_check = rand_seed_check(int(data[_FULL_KEY_RAND_SEED]))
        if self.rand_seed is None:
            self.rand_seed = seed_check
        elif seed_check != self.rand_seed:
            raise ValueError(
                f"Found seed {seed_check} in log file, but file name "
                f"indicates seed {self.rand_seed}.")

        self.__state = (self.__state | 1) & (~4)

    def state_section(self, lines: list[str]) -> None:
        """
        Process the data of the final state section.

        :param lines: the lines of that section
        """
        data: Final[dict[str, str]] = parse_key_values(lines)

        self.total_fes = check_to_int_range(
            data[KEY_TOTAL_FES], KEY_TOTAL_FES, 1,
            1_000_000_000_000_000 if self.max_fes is None else self.max_fes)
        self.total_time_millis = check_to_int_range(
            data[KEY_TOTAL_TIME_MILLIS], KEY_TOTAL_TIME_MILLIS, 0,
            1_000_000_000_000 if self.max_time_millis is None else
            ((1_000_000 + self.max_time_millis) * 1_000))
        if self.max_time_millis is not None:
            _check_max_time_millis(self.max_time_millis, self.total_fes,
                                   self.total_time_millis)
        self.best_f = str_to_num(data[KEY_BEST_F])
        if not isfinite(self.best_f):
            raise ValueError(f"infinite best f detected: {self.best_f}")
        self.last_improvement_fe = check_to_int_range(
            data[KEY_LAST_IMPROVEMENT_FE], KEY_LAST_IMPROVEMENT_FE, 1,
            self.total_fes)
        self.last_improvement_time_millis = check_to_int_range(
            data[KEY_LAST_IMPROVEMENT_TIME_MILLIS],
            KEY_LAST_IMPROVEMENT_TIME_MILLIS, 0, self.total_time_millis)
        self.__state = (self.__state | 2) & (~8)
