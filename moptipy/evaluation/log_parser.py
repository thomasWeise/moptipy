"""A section consumer can load the sections from a log file."""

from abc import ABC
from os import listdir
from os.path import isfile, isdir, join, dirname, basename
from typing import List, Final, Optional

from moptipy.api import logging
from moptipy.utils.console import logger
from moptipy.utils.nputils import rand_seed_check
from moptipy.utils.path import Path
from moptipy.utils.strings import sanitize_name, PART_SEPARATOR
from moptipy.utils.logger import SECTION_START, SECTION_END, COMMENT_CHAR


class LogParser(ABC):
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
        return path.endswith(logging.FILE_SUFFIX)

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
            raise ValueError(f"Title cannot be empty, but is '{title}'.")
        if title.startswith(logging.ERROR_SECTION_PREFIX):
            raise ValueError(f"Encountered error section '{title}'.")
        return False

    # noinspection PyUnusedLocal
    # noinspection PyMethodMayBeStatic
    def lines(self, lines: List[str]) -> bool:
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
        except BaseException as be:
            raise ValueError(f"Error when starting file '{file}'") from be

        if retval:
            if self.__print_file_start:
                logger(f"parsing file '{file}'.")
        else:
            if self.__print_file_start:
                logger(f"skipping file '{file}'.")
            return True

        lines: List[str] = []
        buffer: List[str] = []
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
                    except BaseException as be:
                        raise ValueError(
                            f"Error when reading lines from file '{file}' "
                            f"while in section '{section}'."
                            if state == 1 else
                            f"Error when reading lines from file '{file}'.") \
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
                        ValueError("Line should start with "
                                   f"'{sect_start}' but is "
                                   f"'{orig_cur}' in file '{file}'.")
                    section = cur[len(sect_start):]
                    if len(section) <= 0:
                        ValueError("Section title cannot be empty in "
                                   f"'{file}', but encountered '{orig_cur}'.")
                    state = 1
                    sec_end = sect_end + section
                    wants_section = self.start_section(section)
                elif state == 1:
                    if cur == sec_end:
                        state = 2
                        if wants_section:
                            try:
                                do_next = self.lines(lines)
                            except BaseException as be:
                                raise ValueError(
                                    "Error when processing section "
                                    f"'{section}' in file '{file}'.") from be
                            lines.clear()
                            if not do_next:
                                break
                    elif wants_section:
                        lines.append(cur)

        if state == 0:
            raise ValueError(f"Log file '{file}' contains no section.")
        if state == 1:
            raise ValueError(f"Log file '{file}' ended before"
                             f"encountering '{sec_end}'.")

        try:
            retval = self.end_file()
        except BaseException as be:
            raise ValueError("Error when ending section parsing "
                             f"of file '{file}'.") from be
        if self.__print_file_end:
            logger(f"finished parsing file '{file}'.")
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
                           f"directory '{folder}'.")
            else:
                raise ValueError(
                    f"Depth must be >= 0, but is {self.__depth}??")
        self.__depth += 1

        if not self.start_dir(folder):
            if self.__print_dir_start:
                logger(f"skipping directory '{folder}'.")
            return True
        if self.__print_dir_start:
            logger(f"entering directory '{folder}'.")

        do_files = True
        do_dirs = True
        for sub in listdir(folder):
            sub = Path.path(join(folder, sub))
            if isfile(sub):
                if do_files:
                    if not self.parse_file(sub):
                        logger(f"will parse no more files in '{folder}'.")
                        if not do_dirs:
                            break
                        do_files = False
            elif isdir(sub):
                if do_dirs:
                    if not self.parse_dir(sub):
                        logger("will parse no more sub-directories "
                               f"of '{folder}'.")
                        if not do_files:
                            break
                        do_dirs = False

        retval = self.end_dir(folder)
        if self.__print_dir_end:
            logger(f"finished parsing directory '{folder}'.")

        self.__depth -= 1
        if self.__depth <= 0:
            if self.__depth == 0:
                if self.__print_begin_end:
                    logger(f"finished recursive parsing of "
                           f"directory '{folder}'.")
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
        self.algorithm: Optional[str] = None

        #: The name of the instance to which the current log file belongs.
        self.instance: Optional[str] = None

        #: The random seed of the current log file.
        self.rand_seed: Optional[int] = None

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

        start = f"{self.algorithm}{PART_SEPARATOR}" \
                f"{self.instance}{PART_SEPARATOR}0x"
        base = basename(path)
        if not base.startswith(start):
            raise ValueError(
                f"File name of '{path}' should start with '{start}'.")
        self.rand_seed = rand_seed_check(int(
            base[len(start):(-len(logging.FILE_SUFFIX))], base=16))

        return True

    def end_file(self) -> bool:
        """Finalize parsing a file."""
        self.rand_seed = None
        self.algorithm = None
        self.instance = None
        return super().end_file()
