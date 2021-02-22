from abc import ABC
from io import open
from math import isfinite
from os.path import realpath
from re import sub
from typing import Optional, List, Union

from moptipy.utils import logging


class Logger:
    """
    A class for logging data a text file.
    """

    def __init__(self, path: str):
        """
        Create a new logger.

        :param str path: the path to the log file
        """
        if not isinstance(path, str):
            raise ValueError("Path must be string but is '"
                             + str(type(path)) + "'.")
        path = realpath(path)

        self.__file = open(file=path,
                           mode="wt",
                           encoding="utf-8",
                           errors="strict")

        self.__section = None
        self.__starts_new_line = True
        self.__orig_path = path
        self.__sections = set()

    def __enter__(self):
        return self

    def _error(self, message):
        message = [(f if isinstance(f, str) else "'" + str(f) + "'") for f in message]
        if self.__section is None:
            message[len(message):] = [" in logger '", self.__orig_path, "'."]
        else:
            message[len(message):] = [" in section '", self.__section,
                                      "' of logger '", self.__orig_path + "'."]
        raise ValueError("".join(message))

    def __exit__(self, exception_type, exception_value, traceback):
        if not (self.__section is None):
            self._error(["Cannot close logger, because section still open"])
        if not (self.__file is None):
            if not self.__starts_new_line:
                self.__file.write("\n")
            self.__file.close()
            self.__file = None

    def _open_section(self, title: str) -> None:
        """
        internal method for opening a new section.

        :param str title: the section title
        """
        if self.__file is None:
            self._error(["Cannot open section '",
                         title, "' because logger already closed"])

        if not (self.__section is None):
            self._error(["Cannot open section '",
                         title, "' because another one is open"])

        new_title = title.strip().upper()
        if new_title != title:
            self._error(["Cannot open section '",
                         title, "' because title is invalid"])

        if title in self.__sections:
            self._error(["Section '", title, "' already done"])
        else:
            self.__sections.add(title)

        self.__file.write(logging.SECTION_START + title + "\n")
        self.__closer = logging.SECTION_END + title + "\n"
        self.__starts_new_line = True
        self.__section = title

    def _close_section(self, title: str) -> None:
        """
        internal method for closing a section.

        :param str title: the section title
        """
        if (self.__section is None) or (self.__section != title):
            self._error(["Cannot open section '",
                         title, "' since it is not open"])
        printer = self.__closer
        if not self.__starts_new_line:
            printer = "\n" + printer

        self.__file.write(printer)
        self.__closer = None
        self.__starts_new_line = True
        self.__section = None

    def _comment(self, comment: str) -> None:
        if self.__section is None:
            self._error(["Cannot write if not inside section"])
        self.__file.write(logging.COMMENT_CHAR + " "
                          + sub(r"\s+", " ", comment.strip())
                          + "\n")
        self.__starts_new_line = True

    def _write(self, text: str) -> None:
        """
        internal method for writing a string.

        :param str text: the text to write
        """
        if self.__section is None:
            self._error(["Cannot write if not inside section"])

        if self.__closer in text:
            self._error(["String '", self.__closer,
                         "' must not be contained in output"])

        self.__file.write(text)
        self.__starts_new_line = text.endswith("\n")

    def key_values(self, title: str) -> 'KeyValuesSection':
        """
        Create a log section for key-value pairs.

        :param str title: the title of the new section
        :return: the new logger
        :rtype: KeyValuesSection

        >>> from moptipy.utils.logger import Logger
        >>> from moptipy.utils.temp import TempFile
        >>> with TempFile() as t:
        ...     with Logger(str(t)) as l:
        ...         with l.key_values("B") as kv:
        ...             kv.key_value("a", "b")
        ...             with kv.scope("c") as kvc:
        ...                 kvc.key_value("d", 12)
        ...                 kvc.key_value("e", True)
        ...             kv.key_value("f", 3.3)
        ...     print(open(str(t), "r").read().splitlines())
        ['BEGIN_B', 'a:b', 'c.d:12', 'c.e:True', 'f:3.3', 'f(hex):0x1.a666666666666p+1', 'END_B']
        """
        return KeyValuesSection(title=title, logger=self, prefix="",
                                done=None)

    def csv(self, title: str, header: List[str]) -> 'CsvSection':
        """
        Create a log section for CSV data.

        :param str title: the title of the new section
        :param List[str] header: the list of column titles
        :return: the new logger
        :rtype: CsvSection

        >>> from moptipy.utils.logger import Logger
        >>> from moptipy.utils.temp import TempFile
        >>> with TempFile() as t:
        ...     with Logger(str(t)) as l:
        ...         with l.csv("A", ["x", "y"]) as csv:
        ...             csv.row([1,2])
        ...             csv.row([3,4])
        ...             csv.row([None, 12])
        ...     print(open(str(t), "r").read().splitlines())
        ['BEGIN_A', 'x;y', '1;2', '3;4', ';12', 'END_A']
        """
        return CsvSection(title=title, logger=self, header=header)

    def text(self, title: str) -> 'TextSection':
        """
        Create a log section for unstructured text.

        :param str title: the title of the new section
        :return: the new logger
        :rtype: TextSection

        >>> from moptipy.utils.logger import Logger
        >>> from moptipy.utils.temp import TempFile
        >>> with TempFile() as t:
        ...     with Logger(str(t)) as l:
        ...         with l.text("C") as tx:
        ...             tx.write("aaaaaa")
        ...             tx.write("bbbbb")
        ...             tx.write("\\n")
        ...             tx.write("ccccc")
        ...     print(open(str(t), "r").read().splitlines())
        ['BEGIN_C', 'aaaaaabbbbb', 'ccccc', 'END_C']
        """
        return TextSection(title=title, logger=self)


class _Section(ABC):

    def __init__(self, title: Optional[str], logger: Logger):
        self._logger = logger
        self._title = title
        if not (title is None):
            # noinspection PyProtectedMember
            logger._open_section(title)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if not (self._title is None):
            # noinspection PyProtectedMember
            self._logger._close_section(self._title)
            self._title = None
        self._logger = None

    def comment(self, comment: str) -> None:
        """
        Write a comment line.

        :param str comment: the comment to write
        """
        # noinspection PyProtectedMember
        self._logger._comment(comment)


class CsvSection(_Section):
    """
    A logger that is designed to output CSV data.
    """

    def __init__(self, title: str, logger: Logger, header: List[str]):
        super().__init__(title, logger)

        self.__header_len = len(header)
        if self.__header_len <= 0:
            # noinspection PyProtectedMember
            logger._error(["Empty header ", header,
                           " invalid for a CSV section"])

        for c in header:
            if (not (isinstance(c, str))) or logging.CSV_SEPARATOR in c:
                # noinspection PyProtectedMember
                logger._error(["Invalid column ", c])

        # noinspection PyProtectedMember
        logger._write(logging.CSV_SEPARATOR.join(
            [c.strip() for c in header]) + "\n")

    def row(self, row: List[Union[int, float, bool, None]]) -> None:
        """
        Write a row of csv data.

        :param List[Union[int,float,bool]] row: the row of data
        """
        if self.__header_len != len(row):
            # noinspection PyProtectedMember
            self._logger._error(["Header of CSV section demands ",
                                 str(self.__header_len),
                                 " columns, but row ", row,
                                 " has ", str(len(row))])

        # noinspection PyProtectedMember
        row = ["" if c is None else
               str(c) if (isinstance(c, int) or isinstance(c, bool))
               else (logging.format_float(c) if isinstance(c, float)
                     else self._logger._error(["Invalid log value ", c,
                                               " in row ", row]))
               for c in row]

        # noinspection PyProtectedMember
        self._logger._write(logging.CSV_SEPARATOR.join(row) + "\n")


class KeyValuesSection(_Section):
    """
    A logger for key-value pairs.
    """

    def __init__(self, title: Optional[str], logger: Logger, prefix: str, done):
        super().__init__(title=title, logger=logger)
        self._prefix = prefix
        if done is None:
            self.__done = set(prefix)
        else:
            self.__done = done
            if prefix in done:
                # noinspection PyProtectedMember
                logger._error(["Prefix '", prefix, "' already done"])
            else:
                done.add(prefix)

    def key_value(self, key: str, value,
                  also_hex: bool = False) -> None:
        """
        Write a key-value pair

        :param str key: the key
        :param value: the value
        :param bool also_hex: also store the value as hexadecimal version
        """
        key = self._prefix + logging.sanitize_name(key)
        if key in self.__done:
            # noinspection PyProtectedMember
            self._logger._error(["Key '", key, "' already used"])
        else:
            self.__done.add(key)

        the_hex = None
        if isinstance(value, float):
            txt = logging.format_float(value)
            if isfinite(value):
                if also_hex or (("e" in txt) or ("." in txt)):
                    the_hex = float.hex(value)
        else:
            if isinstance(value, complex):
                txt = logging.format_float(value)
            else:
                txt = str(value)
                if also_hex and isinstance(value, int):
                    the_hex = hex(value)

        txt = logging.KEY_VALUE_SEPARATOR.join([key, txt]) + "\n"

        if the_hex:
            txt += logging.KEY_VALUE_SEPARATOR.join(
                [key + logging.KEY_HEX_VALUE, the_hex]) + "\n"

        # noinspection PyProtectedMember
        self._logger._write(txt)

    def scope(self, prefix: str) -> 'KeyValuesSection':
        """
        Create a new scope for key prefixes

        :param str prefix: the key prefix
        :return: the new logger
        :rtype: KeyValuesSection
        """
        return KeyValuesSection(title=None, logger=self._logger,
                                prefix=((prefix if (self._prefix is None)
                                        else (self._prefix
                                        + logging.sanitize_name(prefix)))
                                        + "."),
                                done=self.__done)


class TextSection(_Section):
    """
    A logger for raw, unprocessed text.
    """

    def __init__(self, title: str, logger: Logger):
        super().__init__(title, logger)

    def write(self, string: str) -> None:
        """
        Write a raw string to the logger.

        :param str string: the string to be written
        """
        # noinspection PyProtectedMember
        self._logger._write(string)
