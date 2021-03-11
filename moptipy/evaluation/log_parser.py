"""A section consumer can load the sections from a log file."""

from abc import ABC
from datetime import datetime
from os import listdir
from os.path import isfile, isdir, join
from typing import List

from moptipy.utils import logging
from moptipy.utils.io import canonicalize_path, enforce_file, enforce_dir


class LogParser(ABC):
    """
    A log parser can parse a log file and separate the sections.

    The log parser is designed to load data from text files generated
    by :class:`~moptipy.utils.logger.FileLogger`. It can also recursively
    parse directories.
    """

    def __init__(self,
                 print_file_start: bool = False,
                 print_file_end: bool = False,
                 print_dir_start: bool = True,
                 print_dir_end=True):
        """
        Initialize the log parser.

        :param bool print_file_start: log to stdout when opening a file
        :param bool print_file_end: log to stdout when closing a file
        :param bool print_dir_start: log to stdout when entering a directory
        :param bool print_dir_end: log to stdout when leaving a directory
        """
        self.__print_file_start = print_file_start
        self.__print_file_end = print_file_end
        self.__print_dir_start = print_dir_start
        self.__print_dir_end = print_dir_end

    # noinspection PyUnusedLocal
    # noinspection PyMethodMayBeStatic
    def start_dir(self, path: str) -> bool:
        """
        Enter a directory to parse all files inside.

        This method is called by :meth:`parse_dir`. If it returns `True`,
        every sub-directory inside of it will be passed to :meth:`start_dir`
        and every file will be passed to :meth:`start_file`.
        Only if `True` is returned, :meth:`end_dir` will be invoked and its
        return value will be the return value of :meth:`parse_dir`. If `False`
        is returned, then :meth:`parse_dir` will return immediately and return
        `True`.

        :param str path: the path of the directory
        :return: `True` if all the files and sub-directories inside the
            directory should be processed, `False` if this directory should
            be skipped and parsing should continue with the next sibling
            directory
        :rtype: bool
        """
        del path
        return True

    # noinspection PyUnusedLocal
    # noinspection PyMethodMayBeStatic
    def end_dir(self, path: str) -> bool:
        """
        Enter a directory to parse all files inside.

        This method is called by :meth:`parse_dir`. If it returns `True`,
        every sub-directory inside of it will be passed to :meth:`start_dir`
        and every file will be passed to :meth:`start_file`.

        :param str path: the path of the directory
        :return: `True` if all the files and sub-directories inside the
            directory should be processed, `False` if this directory should
            be skipped and parsing should continue with the next sibling
            directory
        :rtype: bool
        """
        del path
        return True

    # noinspection PyUnusedLocal
    # noinspection PyMethodMayBeStatic
    def start_file(self, path: str) -> bool:
        """
        Decide whether to parsing a file.

        This method is called by :meth:`parse_file`. If it returns `True`,
        then we will open and parse the file. If it returns `False`, then
        the fill will not be parsed and :meth:`parse_file` will return
        `True` immediately.

        :param str path: the file path
        :return: `True` if the file should be parsed, `False` if it should be
            skipped (and :meth:`parse_file` should return `True`).
        :rtype: bool
        """
        del path
        return True

    # noinspection PyUnusedLocal
    # noinspection PyMethodMayBeStatic
    def start_section(self, title: str) -> bool:
        """
        Start a section.

        If this method returns `True`, then all the lines of text of the
        section `title` will be read and together passed to :meth:`lines`.
        If this method returns `False`, then the section will be skipped
        and we fast-forward to the next section, if any, or to the call
        of :meth:`end_file`.

        :param str title: the section title
        :return: `True` if the section data should be loaded and passed to
            :meth:`lines`, `False` of the section can be skipped. In that
            case, we will fast forward to the next :meth:`start_section`.
        :rtype: bool
        """
        del title
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
        parsing process (and call :meth:`end_file`) if there is no more
        section in the file.

        :param List[str] lines: the line to consume
        :return: `True` if further parsing is necessary and the next section
            should be fed to :meth:`start_section`, `False` if the parsing
            process can be terminated, in which case we will fast forward to
            :meth:`end_file`
        :rtype: bool
        """
        del lines
        return True

    # noinspection PyMethodMayBeStatic
    def end_file(self) -> bool:
        """
        End a file.

        This method is invoked when we have reached the end of the current
        file. Its return value, `True` or `False`, will then be returned
        by :meth:`parse_file`, which is the entry point for the file parsing
        process.

        :return: the return value to be returned by :meth:`parse_file`
        :rtype: bool
        """
        return True

    def parse_file(self, path: str) -> bool:
        """
        Parse the contents of a file.

        This method first calls the function :meth:`start_file` to see
        whether the file should be parsed. If :meth:`start_file` returns
        `True`, then the file is parsed. If :meth:`start_file` returns
        `False`, then this method returns `False` directly.
        If the file is parsed, then :meth:`start_sections` will be invoked for
        each section (until the parsing is finished) and :meth:`lines` for
        each section content (if requested). At the end, :meth:`end_file` is
        invoked.

        This method can either be called directly or is called by
        :meth:`parse_dir`. In the latter case, if :meth:`parse_file` returned
        `True`, the next file in the current directory will be parsed. If it
        returns `False`, then no file located in the current directory will be
        parsed, while other directories and/or sub-directories will still be
        processed.

        :param str path: the file to parse
        :return: the return value of :meth:`end_file`
        :rtype: bool
        """
        path = enforce_file(canonicalize_path(path))

        retval: bool
        try:
            retval = self.start_file(path)
        except BaseException as be:
            raise ValueError(f"Error when starting file '{path}'") from be

        if retval:
            if self.__print_file_start:
                print(f"{datetime.now()}: parsing file {path}.")
        else:
            if self.__print_file_start:
                print(f"{datetime.now()}: skipping file {path}.")
            return True

        lines: List[str] = list()
        buffer: List[str] = []
        state: int = 0
        wants_section: bool = False
        sec_end: str = ""
        section: str = ""

        index: int = 0
        with open(path, "rt") as handle:
            while True:

                # get the next line
                if index >= len(buffer):
                    try:
                        buffer = handle.readlines(128)
                    except BaseException as be:
                        raise ValueError(
                            f"Error when reading lines from file '{path}' "
                            f"while in section '{section}'."
                            if state == 1 else
                            f"Error when reading lines from file '{path}'.") \
                            from be
                    if buffer is None:
                        break
                    index = 0

                orig_cur = buffer[index]
                index += 1

                # strip next line from comments and white space
                cur = orig_cur.strip()
                if len(cur) <= 0:
                    continue

                i = cur.find(logging.COMMENT_CHAR)
                if i >= 0:
                    cur = cur[:i].strip()
                    if len(cur) <= 0:
                        continue

                if state in (0, 2):
                    if not cur.startswith(logging.SECTION_START):
                        ValueError("Line should start with "
                                   f"'{logging.SECTION_START}' but is "
                                   f"'{orig_cur}' in file '{path}'.")
                    section = cur[len(logging.SECTION_START):]
                    if len(section) <= 0:
                        ValueError("Section title cannot be empty in "
                                   f"'{path}', but encountered '{orig_cur}'.")
                    state = 1
                    sec_end = logging.SECTION_END + section
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
                                    f"'{section}' in file '{path}'.") from be
                            lines.clear()
                            if not do_next:
                                break
                    elif wants_section:
                        lines.append(cur)

        if state == 0:
            raise ValueError(f"Log file '{path}' contains no section.")
        if state == 1:
            raise ValueError(f"Log file '{path}' ended before"
                             f"encountering '{sec_end}'.")

        try:
            retval = self.end_file()
        except BaseException as be:
            raise ValueError("Error when ending section parsing "
                             f"of file '{path}'.") from be
        if self.__print_file_end:
            print(f"{datetime.now()}: finished parsing file {path}.")
        return retval

    def parse_dir(self, path: str) -> bool:
        """
        Recursively parse the given directory.

        :param str path: the path to the directory
        :return: `True` either if :meth:`start_dir` returned `False` or
            :meth:`end_dir` returned `True`, `False` otherwise
        :rtype: bool
        """
        path = enforce_dir(canonicalize_path(path))

        if not self.start_dir(path):
            if self.__print_dir_start:
                print(f"{datetime.now()}: skipping directory '{path}'.")
            return True
        if self.__print_dir_start:
            print(f"{datetime.now()}: entering directory '{path}'.")

        do_files = True
        do_dirs = True
        for sub in listdir(path):
            sub = join(path, sub)
            if isfile(sub):
                if do_files:
                    if not self.parse_file(sub):
                        print(f"{datetime.now()}: will parse no more "
                              f"files in '{path}'.")
                        if not do_dirs:
                            break
                        do_files = False
            elif isdir(sub):
                if do_dirs:
                    if not self.parse_dir(sub):
                        print(f"{datetime.now()}: will parse no more "
                              f"sub-directories of '{path}'.")
                        if not do_files:
                            break
                        do_dirs = False

        retval = self.end_dir(path)
        if self.__print_dir_end:
            print(f"{datetime.now()}: finished parsing directory '{path}'.")
        return retval
