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
:meth:`~moptipy.evaluation.end_results.EndResult.from_logs` can load
:class:`~moptipy.evaluation.end_results.EndResult` records from the logs
and the method :meth:`~moptipy.evaluation.progress.Progress.from_logs` in
module :mod:`~moptipy.evaluation.progress` reads the whole
:class:`~moptipy.evaluation.progress.Progress` that the algorithms make
over time.
"""

from math import isfinite
from os import listdir
from os.path import basename, dirname, isdir, isfile, join
from typing import Final

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
    SCOPE_PROCESS,
    SECTION_FINAL_STATE,
    SECTION_SETUP,
)
from moptipy.evaluation._utils import _check_max_time_millis
from moptipy.utils.console import logger
from moptipy.utils.logger import (
    COMMENT_CHAR,
    SCOPE_SEPARATOR,
    SECTION_END,
    SECTION_START,
    parse_key_values,
)
from moptipy.utils.nputils import rand_seed_check
from moptipy.utils.path import Path
from moptipy.utils.strings import (
    PART_SEPARATOR,
    sanitize_name,
    str_to_intfloat,
)
from moptipy.utils.types import check_to_int_range

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


class LogParser:
    """
    A log parser can parse a log file and separate the sections.

    The log parser is designed to load data from text files generated
    by :class:`~moptipy.utils.logger.FileLogger`. It can also recursively
    parse directories.
    """

    def __init__(self,
                 print_begin_end: bool = True,
                 print_file_start: bool = False,
                 print_file_end: bool = False,
                 print_dir_start: bool = True,
                 print_dir_end: bool = True):
        """
        Initialize the log parser.

        :param print_begin_end: log to stdout when starting and ending
            a recursive directory parsing process
        :param print_file_start: log to stdout when opening a file
        :param print_file_end: log to stdout when closing a file
        :param print_dir_start: log to stdout when entering a directory
        :param print_dir_end: log to stdout when leaving a directory
        """
        self.__print_begin_end: Final[bool] = print_begin_end
        self.__print_file_start: Final[bool] = print_file_start
        self.__print_file_end: Final[bool] = print_file_end
        self.__print_dir_start: Final[bool] = print_dir_start
        self.__print_dir_end: Final[bool] = print_dir_end
        self.__depth: int = 0

    # noinspection PyUnusedLocal
    # noinspection PyMethodMayBeStatic
    def start_dir(self, path: Path) -> bool:
        """
        Enter a directory to parse all files inside.

        This method is called by
        :meth:`~moptipy.evaluation.log_parser.LogParser.parse_dir`.
        If it returns `True`, every sub-directory inside of it will be passed
        to :meth:`~moptipy.evaluation.log_parser.LogParser.start_dir`
        and every file will be passed to
        :meth:`~moptipy.evaluation.log_parser.LogParser.start_file`.
        Only if `True` is returned,
        :meth:`~moptipy.evaluation.log_parser.LogParser.end_dir` will be
        invoked and its return value will be the return value of
        :meth:`~moptipy.evaluation.log_parser.LogParser.parse_dir`. If `False`
        is returned, then
        :meth:`~moptipy.evaluation.log_parser.LogParser.parse_dir` will return
        immediately and return `True`.

        :param path: the path of the directory
        :return: `True` if all the files and sub-directories inside the
            directory should be processed, `False` if this directory should
            be skipped and parsing should continue with the next sibling
            directory
        """
        del path
        return True

    # noinspection PyUnusedLocal
    # noinspection PyMethodMayBeStatic
    def end_dir(self, path: Path) -> bool:
        """
        Enter a directory to parse all files inside.

        This method is called by
        :meth:`~moptipy.evaluation.log_parser.LogParser.parse_dir`.
        If it returns `True`, every sub-directory inside of it will be passed
        to :meth:`~moptipy.evaluation.log_parser.LogParser.start_dir`
        and every file will be passed to
        :meth:`~moptipy.evaluation.log_parser.LogParser.start_file`.

        :param path: the path of the directory
        :return: `True` if all the files and sub-directories inside the
            directory should be processed, `False` if this directory should
            be skipped and parsing should continue with the next sibling
            directory
        """
        del path
        return True

    # noinspection PyUnusedLocal
    # noinspection PyMethodMayBeStatic
    def start_file(self, path: Path) -> bool:
        """
        Decide whether to start parsing a file.

        This method is called by
        :meth:`~moptipy.evaluation.log_parser.LogParser.parse_file`. If it
        returns `True`, then we will open and parse the file. If it returns
        `False`, then the fill will not be parsed and
        :meth:`~moptipy.evaluation.log_parser.LogParser.parse_file` will
        return `True` immediately.

        :param path: the file path
        :return: `True` if the file should be parsed, `False` if it should be
            skipped (and
            :meth:`~moptipy.evaluation.log_parser.LogParser.parse_file` should
            return `True`).
        """
        return path.endswith(FILE_SUFFIX)

    # noinspection PyMethodMayBeStatic
    def start_section(self, title: str) -> bool:
        """
        Start a section.

        If this method returns `True`, then all the lines of text of the
        section `title` will be read and together passed to
        :meth:`~moptipy.evaluation.log_parser.LogParser.lines`.
        If this method returns `False`, then the section will be skipped
        and we fast-forward to the next section, if any, or to the call
        of :meth:`~moptipy.evaluation.log_parser.LogParser.end_file`.

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
        parsing process (and call
        :meth:`~moptipy.evaluation.log_parser.LogParser.end_file`) if there is
        no more section in the file.

        :param lines: the lines to consume
        :return: `True` if further parsing is necessary and the next section
            should be fed to
            :meth:`~moptipy.evaluation.log_parser.LogParser.start_section`,
            `False` if the parsing
            process can be terminated, in which case we will fast-forward to
            :meth:`~moptipy.evaluation.log_parser.LogParser.end_file`
        """
        del lines
        return True

    # noinspection PyMethodMayBeStatic
    def end_file(self) -> bool:
        """
        End a file.

        This method is invoked when we have reached the end of the current
        file. Its return value, `True` or `False`, will then be returned
        by :meth:`~moptipy.evaluation.log_parser.LogParser.parse_file`, which
        is the entry point for the file parsing process.

        :return: the return value to be returned by
            :meth:`~moptipy.evaluation.log_parser.LogParser.parse_file`
        """
        return True

    def parse_file(self, path: str) -> bool:
        """
        Parse the contents of a file.

        This method first calls the function
        :meth:`~moptipy.evaluation.log_parser.LogParser.start_file` to see
        whether the file should be parsed. If
        :meth:`~moptipy.evaluation.log_parser.LogParser.start_file` returns
        `True`, then the file is parsed. If
        :meth:`~moptipy.evaluation.log_parser.LogParser.start_file` returns
        `False`, then this method returns `False` directly.
        If the file is parsed, then
        :meth:`~moptipy.evaluation.log_parser.LogParser.start_section` will
        be invoked for each section (until the parsing is finished) and
        :meth:`~moptipy.evaluation.log_parser.LogParser.lines` for
        each section content (if requested). At the end,
        :meth:`~moptipy.evaluation.log_parser.LogParser.end_file` is invoked.

        This method can either be called directly or is called by
        :meth:`~moptipy.evaluation.log_parser.LogParser.parse_dir`. In the
        latter case, if
        :meth:`~moptipy.evaluation.log_parser.LogParser.parse_file` returned
        `True`, the next file in the current directory will be parsed. If it
        returns `False`, then no file located in the current directory will be
        parsed, while other directories and/or sub-directories will still be
        processed.

        :param path: the file to parse
        :return: the return value received from invoking
            :meth:`~moptipy.evaluation.log_parser.LogParser.end_file`
        """
        file: Final[Path] = Path.file(path)

        retval: bool
        try:
            retval = self.start_file(file)
        except Exception as be:
            raise ValueError(f"Error when starting file {file!r}") from be

        if retval:
            if self.__print_file_start:
                logger(f"parsing file {file!r}.")
        else:
            if self.__print_file_start:
                logger(f"skipping file {file!r}.")
            return True

        lines: list[str] = []
        buffer: list[str] = []
        state: int = 0
        wants_section: bool = False
        sec_end: str = ""
        section: str = ""
        sect_start: Final[str] = SECTION_START
        sect_end: Final[str] = SECTION_END
        cmt_chr: Final[str] = COMMENT_CHAR

        index: int = 0
        with file.open_for_read() as handle:
            while True:

                # get the next line
                if index >= len(buffer):
                    try:
                        buffer = handle.readlines(128)
                    except Exception as be:
                        raise ValueError(
                            f"Error when reading lines from file {file!r} "
                            f"while in section {section!r}."
                            if state == 1 else
                            f"Error when reading lines from file {file!r}.") \
                            from be
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
                                         f"{orig_cur!r} in file {file!r}.")
                    section = cur[len(sect_start):]
                    if len(section) <= 0:
                        raise ValueError(
                            "Section title cannot be empty in "
                            f"{file!r}, but encountered {orig_cur!r}.")
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
                                    f"{section!r} in file {file!r}.") from be
                            lines.clear()
                            if not do_next:
                                break
                    elif wants_section:
                        lines.append(cur)

        if state == 0:
            raise ValueError(f"Log file {file!r} contains no section.")
        if state == 1:
            raise ValueError(f"Log file {file!r} ended before"
                             f"encountering {sec_end!r}.")

        try:
            retval = self.end_file()
        except Exception as be:
            raise ValueError("Error when ending section parsing "
                             f"of file {file!r}.") from be
        if self.__print_file_end:
            logger(f"finished parsing file {file!r}.")
        return retval

    def parse_dir(self, path: str) -> bool:
        """
        Recursively parse the given directory.

        :param path: the path to the directory
        :return: `True` either if
            :meth:`~moptipy.evaluation.log_parser.LogParser.start_dir`
            returned `False` or
            :meth:`~moptipy.evaluation.log_parser.LogParser.end_dir` returned
            `True`, `False` otherwise
        """
        folder: Final[Path] = Path.directory(path)

        if self.__depth <= 0:
            if self.__depth == 0:
                if self.__print_begin_end:
                    logger("beginning recursive parsing of "
                           f"directory {folder!r}.")
            else:
                raise ValueError(
                    f"Depth must be >= 0, but is {self.__depth}??")
        self.__depth += 1

        if not self.start_dir(folder):
            if self.__print_dir_start:
                logger(f"skipping directory {folder!r}.")
            return True
        if self.__print_dir_start:
            logger(f"entering directory {folder!r}.")

        do_files = True
        do_dirs = True
        for subpath in listdir(folder):
            sub = Path.path(join(folder, subpath))
            if isfile(sub):
                if do_files and (not self.parse_file(sub)):
                    logger(f"will parse no more files in {folder!r}.")
                    if not do_dirs:
                        break
                    do_files = False
            elif isdir(sub) and do_dirs and (not self.parse_dir(sub)):
                logger("will parse no more sub-directories "
                       f"of {folder!r}.")
                if not do_files:
                    break
                do_dirs = False

        retval = self.end_dir(folder)
        if self.__print_dir_end:
            logger(f"finished parsing directory {folder!r}.")

        self.__depth -= 1
        if self.__depth <= 0:
            if self.__depth == 0:
                if self.__print_begin_end:
                    logger(f"finished recursive parsing of "
                           f"directory {folder!r}.")
            else:
                raise ValueError(
                    f"Depth must be >= 0, but is {self.__depth}??")

        return retval

    def parse(self, path: str) -> bool:
        """
        Parse either a directory or a file.

        If `path` identifies a file,
        :meth:`~moptipy.evaluation.log_parser.LogParser.parse_file` is invoked
        and its result is returned. If `path` identifies a directory, then
        :meth:`~moptipy.evaluation.log_parser.LogParser.parse_dir` is invoked
        and its result is returned.

        :param path: a path identifying either a directory or a file.
        :return: the result of the appropriate parsing routing
        :raises ValueError: if `path` does not identify a directory or file
        """
        npath = Path.path(path)
        if isfile(npath):
            return self.parse_file(npath)
        if isdir(npath):
            return self.parse_dir(npath)
        raise ValueError(
            f"Path {npath} is neither a file nor a directory?")


class ExperimentParser(LogParser):
    """A log parser following our pre-defined experiment structure."""

    def __init__(self):
        """Initialize the experiment parser."""
        super().__init__(print_begin_end=True, print_dir_start=True)

        #: The name of the algorithm to which the current log file belongs.
        self.algorithm: str | None = None

        #: The name of the instance to which the current log file belongs.
        self.instance: str | None = None

        #: The random seed of the current log file.
        self.rand_seed: int | None = None

    def start_file(self, path: Path) -> bool:
        """
        Decide whether to start parsing a file and setup meta-data.

        :param path: the file path
        :return: `True` if the file should be parsed, `False` if it should be
            skipped (and
            :meth:`~moptipy.evaluation.log_parser.LogParser.parse_file` should
            return `True`).
        """
        if not super().start_file(path):
            return False

        inst_dir = dirname(path)
        algo_dir = dirname(inst_dir)
        self.instance = sanitize_name(basename(inst_dir))
        self.algorithm = sanitize_name(basename(algo_dir))

        start = (f"{self.algorithm}{PART_SEPARATOR}"
                 f"{self.instance}{PART_SEPARATOR}0x")
        base = basename(path)
        if (not base.startswith(start)) and \
                (not base.casefold().startswith(start.casefold())):
            # case-insensitive comparison needed because of Windows
            raise ValueError(
                f"File name of {path!r} should start with {start!r}.")
        self.rand_seed = rand_seed_check(int(
            base[len(start):(-len(FILE_SUFFIX))], base=16))

        return True

    def end_file(self) -> bool:
        """Finalize parsing a file."""
        self.rand_seed = None
        self.algorithm = None
        self.instance = None
        return super().end_file()


class SetupAndStateParser(ExperimentParser):
    """
    A log parser which loads and processes the basic data from the logs.

    This parser processes the `SETUP` and `STATE` sections of a log file and
    stores the performance-related information in member variables.
    """

    def __init__(self):
        """Create the basic data parser."""
        super().__init__()
        self.total_fes: int | None = None
        self.total_time_millis: int | None = None
        self.best_f: int | float | None = None
        self.last_improvement_fe: int | None = None
        self.last_improvement_time_millis: int | None = None
        self.goal_f: int | float | None = None
        self.max_fes: int | None = None
        self.max_time_millis: int | None = None
        self.__state: int = 0

    def start_file(self, path: Path) -> bool:
        """
        Begin parsing the file identified by `path`.

        :param path: the path identifying the file
        """
        if not super().start_file(path):
            return False
        if self.__state != 0:
            raise ValueError(f"Illegal state when trying to parse {path}.")
        return True

    def process(self) -> None:
        """
        Process the result of the log parsing.

        This function is invoked by :meth:`end_file` if the end of the parsing
        process is reached. By now, all the data should have been loaded and it
        can be passed on to wherever it should be passed to.
        """

    def end_file(self) -> bool:
        """
        Finalize parsing a file and invoke the :meth:`process` method.

        This method invokes the :meth:`process` method to process the parsed
        data.

        :returns: `True` if parsing should be continued, `False` otherwise
        """
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
        self.process()  # invoke the handler
        # now clear the data
        self.total_fes = None
        self.total_time_millis = None
        self.best_f = None
        self.last_improvement_fe = None
        self.last_improvement_time_millis = None
        self.goal_f = None
        self.max_fes = None
        self.max_time_millis = None
        self.__state = 0
        return super().end_file()

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
            self.setup_section(lines)
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

    def setup_section(self, lines: list[str]) -> None:
        """
        Parse the data from the `setup` section.

        :param lines: the lines of the section
        """
        data: Final[dict[str, str]] = parse_key_values(lines)
        if _FULL_KEY_GOAL_F in data:
            goal_f = data[_FULL_KEY_GOAL_F]
            if ("e" in goal_f) or ("E" in goal_f) or ("." in goal_f):
                self.goal_f = float(goal_f)
            elif goal_f == "-inf":
                self.goal_f = None
            else:
                self.goal_f = int(goal_f)
        else:
            self.goal_f = None

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
        self.best_f = str_to_intfloat(data[KEY_BEST_F])
        if not isfinite(self.best_f):
            raise ValueError(f"infinite best f detected: {self.best_f}")
        self.last_improvement_fe = check_to_int_range(
            data[KEY_LAST_IMPROVEMENT_FE], KEY_LAST_IMPROVEMENT_FE, 1,
            self.total_fes)
        self.last_improvement_time_millis = check_to_int_range(
            data[KEY_LAST_IMPROVEMENT_TIME_MILLIS],
            KEY_LAST_IMPROVEMENT_TIME_MILLIS, 0, self.total_time_millis)
        self.__state = (self.__state | 2) & (~8)
