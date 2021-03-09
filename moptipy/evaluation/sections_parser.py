"""A section consumer can load the sections from a log file."""

from abc import ABC
from typing import List

from moptipy.utils import logging
from moptipy.utils.io import canonicalize_path, enforce_file


class SectionsParser(ABC):
    """A section consumer can parse a log file and separate the sections."""

    # noinspection PyUnusedLocal
    # noinspection PyMethodMayBeStatic
    def start_dir(self, path: str) -> bool:
        """
        Enter a directory to parse all files inside.

        :param str path: the path of the directory
        :return: `True` if all the files and sub-directories inside the
            directory should be processed, `False` otherwise
        :rtype: bool
        """
        del path
        return True

    # noinspection PyUnusedLocal
    # noinspection PyMethodMayBeStatic
    def start_file(self, path: str) -> bool:
        """
        Start parsing a file.

        :param str path: the file path
        :return: `True` if the file should be parsed, `False` if it should be
            skipped and :meth:`parse` should return `True`.
        :rtype: bool
        """
        del path
        return True

    # noinspection PyUnusedLocal
    # noinspection PyMethodMayBeStatic
    def start_section(self, title: str) -> bool:
        """
        Start a section.

        :param title: the section title
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

        Each line is stripped from whitespace and comments, empty lines are
        omitted.

        :param List[str] lines: the line to consume
        :return: `True` if further parsing is necessary and the next section
            should be fed to :meth:`start_section`, `False` if the parsing
            process can be terminated
        :rtype: bool
        """
        del lines
        return True

    # noinspection PyMethodMayBeStatic
    def end_file(self) -> bool:
        """
        End a file.

        :return: the return value to be returned by :meth:`parse_file`
        :rtype: bool
        """
        return True

    # noinspection PyMethodMayBeStatic
    def end_dir(self) -> bool:
        """
        End parsing a directory.

        :return: `True` or `False`
        :rtype: bool
        """
        return True

    def parse_file(self, file: str) -> bool:
        """
        Parse a file with the given instance of :class:`Sections`.

        :param str file: the file to parse
        :return: the return value of :meth:`end_file`
        :rtype: bool
        """
        file = enforce_file(canonicalize_path(file))

        try:
            if not self.start_file(file):
                return True
        except BaseException as be:
            raise ValueError(f"Error when starting file '{file}'") from be

        lines: List[str] = list()
        buffer: List[str] = []
        state: int = 0
        wants_section: bool = False
        sec_end: str = ""
        section: str = ""

        index: int = 0
        with open(file, "rt") as handle:
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
                            f"Error when reading lines from file '{file}'.")\
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
                                   f"'{orig_cur}' in file '{file}'.")
                    section = cur[len(logging.SECTION_START):]
                    if len(section) <= 0:
                        ValueError("Section title cannot be empty in "
                                   f"'{file}', but encountered '{orig_cur}'.")
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
            return self.end_file()
        except BaseException as be:
            raise ValueError("Error when ending section parsing "
                             f"of file '{file}'.") from be
