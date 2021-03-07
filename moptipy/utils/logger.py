"""Classes for writing structured log files."""
from abc import ABC
from io import open, StringIO
from math import isfinite
from os.path import realpath
from re import sub
from typing import Optional, List, Union, cast

from moptipy.utils import logging
from moptipy.utils.cache import is_new


class Logger(ABC):
    """
    An abstract base class for logging data in a structured way.

    There are two implementations of this, :class:`InMemoryLogger`, which logs
    data in memory and is mainly there for testing and debugging, an
    :class:`FileLogger` which logs to a text file and is to be used in
    experiments with `moptipy`.
    """

    def __init__(self, stream, name: str) -> None:
        """
        Create a new logger.

        :param stream: the stream to which we will log, will be closed when
        the logger is closed
        :param name: the name of the logger
        """
        if not isinstance(name, str):
            raise ValueError("Name must be string but is '"
                             + str(type(name)) + "'.")
        if stream is None:
            raise ValueError("stream must be valid strean but is None.")

        self._stream = stream
        self.__section: Optional[str] = None
        self.__starts_new_line = True
        self.__log_name = name
        self.__sections = is_new()
        self.__closer: Optional[str] = None

    def __enter__(self):
        """
        Enter the logger in a `with` statement.

        :return: `self`
        """
        return self

    def _error(self, message: List) -> None:
        """
        Internal method for raising an :class:`ValueError` with context infos.

        :param List message: the message elements to merge
        :raises ValueError: an error with the message and some context
        information
        """
        message = [(f if isinstance(f, str) else "'" + str(f) + "'")
                   for f in message]
        if self.__section is None:
            message[len(message):] = [" in logger '", self.__log_name, "'."]
        else:
            message[len(message):] = [" in section '", self.__section,
                                      "' of logger '", self.__log_name + "'."]
        raise ValueError("".join(message))

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        """
        Close the logger after leaving the `with` statement.

        :param exception_type: ignored
        :param exception_value: ignored
        :param traceback: ignored
        """
        if not (self.__section is None):
            self._error(["Cannot close logger, because section still open"])
        if not (self._stream is None):
            if not self.__starts_new_line:
                self._stream.write("\n")
            self._stream.close()
            self._stream = None

    def _open_section(self, title: str) -> None:
        """
        An internal method for opening a new section.

        :param str title: the section title
        """
        if self._stream is None:
            self._error(["Cannot open section '",
                         title, "' because logger already closed"])

        if not (self.__section is None):
            self._error(["Cannot open section '",
                         title, "' because another one is open"])

        new_title = title.strip().upper()
        if new_title != title:
            self._error(["Cannot open section '",
                         title, "' because title is invalid"])

        if not self.__sections(title):
            self._error(["Section '", title, "' already done"])

        self._stream.write(logging.SECTION_START + title + "\n")
        self.__closer = logging.SECTION_END + title + "\n"
        self.__starts_new_line = True
        self.__section = title

    def _close_section(self, title: str) -> None:
        """
        An internal method for closing a section.

        :param str title: the section title
        """
        if (self.__section is None) or (self.__section != title):
            self._error(["Cannot open section '",
                         title, "' since it is not open"])
        printer = self.__closer
        if not self.__starts_new_line:
            printer = "\n" + printer

        self._stream.write(printer)
        self.__closer = None
        self.__starts_new_line = True
        self.__section = None

    def _comment(self, comment: str) -> None:
        """
        An internal method for writing a comment.

        :param str comment: the comment
        """
        if self.__section is None:
            self._error(["Cannot write if not inside section"])
        if len(comment) <= 0:
            return
        self._stream.write(logging.COMMENT_CHAR + " "
                           + sub(r"\s+", " ", comment.strip())
                           + "\n")
        self.__starts_new_line = True

    def _write(self, text: str) -> None:
        """
        Internal method for writing a string.

        :param str text: the text to write
        """
        if self.__section is None:
            self._error(["Cannot write if not inside section"])

        if len(text) <= 0:
            return

        if self.__closer in text:
            self._error(["String '", self.__closer,
                         "' must not be contained in output"])

        self._stream.write(text)
        self.__starts_new_line = text.endswith("\n")

    def key_values(self, title: str) -> 'KeyValueSection':
        r"""
        Create a log section for key-value pairs.

        The contents of such a section will be valid YAML mappings, i.w.,
        conform to
        https://yaml.org/spec/1.2/spec.html#mapping//.
        This means they can be parsed with a YAML parser (after removing the
        section start and end marker, of course).

        :param str title: the title of the new section
        :return: the new logger
        :rtype: KeyValueSection

        >>> from moptipy.utils.io import TempFile
        >>> with TempFile() as t:
        ...     with FileLogger(str(t)) as l:
        ...         with l.key_values("B") as kv:
        ...             kv.key_value("a", "b")
        ...             with kv.scope("c") as kvc:
        ...                 kvc.key_value("d", 12)
        ...                 kvc.key_value("e", True)
        ...             kv.key_value("f", 3)
        ...     text = open(str(t), "r").read().splitlines()
        >>> print(text)
        ['BEGIN_B', 'a: b', 'c.d: 12', 'c.e: T', 'f: 3', 'END_B']
        >>> import yaml
        >>> dic = yaml.safe_load("\n".join(text[1:5]))
        >>> print(list(dic.keys()))
        ['a', 'c.d', 'c.e', 'f']
        """
        return KeyValueSection(title=title, logger=self, prefix="",
                               done=None)

    def csv(self, title: str, header: List[str]) -> 'CsvSection':
        """
        Create a log section for CSV data with `;` as column separator.

        The first line will be the headline with the column names.

        :param str title: the title of the new section
        :param List[str] header: the list of column titles
        :return: the new logger
        :rtype: CsvSection

        >>> from moptipy.utils.logger import FileLogger
        >>> from moptipy.utils.io import TempFile
        >>> with TempFile() as t:
        ...     with FileLogger(str(t)) as l:
        ...         with l.csv("A", ["x", "y"]) as csv:
        ...             csv.row([1,2])
        ...             csv.row([3,4])
        ...             csv.row([None, 12])
        ...     text = open(str(t), "r").read().splitlines()
        ...     print(text)
        ['BEGIN_A', 'x;y', '1;2', '3;4', ';12', 'END_A']
        >>> import csv
        >>> for r in csv.reader(text[1:5], delimiter=";"):
        ...     print(r)
        ['x', 'y']
        ['1', '2']
        ['3', '4']
        ['', '12']
        """
        return CsvSection(title=title, logger=self, header=header)

    def text(self, title: str) -> 'TextSection':
        r"""
        Create a log section for unstructured text.

        :param str title: the title of the new section
        :return: the new logger
        :rtype: TextSection

        >>> from moptipy.utils.logger import InMemoryLogger
        >>> from moptipy.utils.io import TempFile
        >>> with InMemoryLogger() as l:
        ...     with l.text("C") as tx:
        ...         tx.write("aaaaaa")
        ...         tx.write("bbbbb")
        ...         tx.write("\n")
        ...         tx.write("ccccc")
        ...     print(l.get_log())
        ['BEGIN_C', 'aaaaaabbbbb', 'ccccc', 'END_C']
        """
        return TextSection(title=title, logger=self)


class FileLogger(Logger):
    """A logger logging to a file."""

    def __init__(self, path: str) -> None:
        """
        Initialize the logger.

        :param str path: the path to the file to open
        """
        if not isinstance(path, str):
            raise ValueError("Path must be string but is '"
                             + str(type(path)) + "'.")
        name = path
        path = realpath(path)
        super().__init__(stream=open(file=path, mode="wt",
                                     encoding="utf-8",
                                     errors="strict"),
                         name=name)


class InMemoryLogger(Logger):
    """A logger logging to a string in memory."""

    def __init__(self) -> None:
        """Initialize the logger."""
        super().__init__(stream=StringIO(),
                         name="in-memory-logger")

    def get_log(self) -> List[str]:
        """
        Obtain all the lines logged to this logger.

        :return: a list of strings with the logged lines
        :rtype: List[str]
        """
        return self._stream.getvalue().splitlines()


class _Section(ABC):

    def __init__(self, title: Optional[str], logger: Logger) -> None:
        """
        Internal constructor. Do not call directly.

        :param Optional[str] title: the section title
        :param Logger logger: the logger
        """
        self._logger = logger
        self._title = title
        if not (title is None):
            # noinspection PyProtectedMember
            logger._open_section(title)

    def __enter__(self):
        """
        The enter method needed for the `with` statement.

        :return: `self`
        """
        return self

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        """
        The exit method needed for the `with` statement.

        :param exception_type: ignored
        :param exception_value: ignored
        :param traceback: ignored
        :return: ignored
        """
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
    """A logger that is designed to output CSV data."""

    def __init__(self, title: str, logger: Logger, header: List[str]) -> None:
        """
        Internal constructor. Do not call directly.

        :param str title: the title
        :param Logger logger: the owning logger
        :param List[str] header: the header
        """
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
        txt = ["" if c is None else
               str(c) if isinstance(c, int)
               else logging.bool_to_str(c) if isinstance(c, bool)
               else (logging.float_to_str(c) if isinstance(c, float) else
                     cast(None, self._logger._error(["Invalid log value ",
                                                     c, " in row ", row])))
               for c in row]

        # noinspection PyProtectedMember
        self._logger._write(logging.CSV_SEPARATOR.join(txt) + "\n")


class KeyValueSection(_Section):
    """A logger for key-value pairs."""

    def __init__(self, title: Optional[str],
                 logger: Logger, prefix: str, done) -> None:
        """
        Internal constructor, do not call directly.

        :param Optional[str] title: the section title, or `None` for nested
        scopes.
        :param Logger logger: the owning logger
        :param str prefix: the prefix
        :param done: the set of already done keys and prefixes
        """
        if not isinstance(prefix, str):
            raise TypeError("prefix must be str but is "
                            + str(type(prefix)))
        super().__init__(title=title, logger=logger)
        self._prefix = prefix
        if done is None:
            self.__done = is_new()
            self.__done(prefix)
        else:
            self.__done = done
            if not done(prefix):
                # noinspection PyProtectedMember
                logger._error(["Prefix '", prefix, "' already done"])

    def key_value(self, key: str, value,
                  also_hex: bool = False) -> None:
        """
        Write a key-value pair.

        :param str key: the key
        :param value: the value
        :param bool also_hex: also store the value as hexadecimal version
        """
        key = self._prefix + logging.sanitize_name(key)
        if not self.__done(key):
            # noinspection PyProtectedMember
            self._logger._error(["Key '", key, "' already used"])

        the_hex = None
        if isinstance(value, float):
            txt = logging.float_to_str(value)
            if isfinite(value):
                if also_hex or (("e" in txt) or ("." in txt)):
                    the_hex = float.hex(value)
        elif isinstance(value, bool):
            txt = logging.bool_to_str(value)
        elif isinstance(value, complex):
            txt = logging.complex_to_str(value)
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

    def scope(self, prefix: str) -> 'KeyValueSection':
        """
        Create a new scope for key prefixes.

        :param str prefix: the key prefix
        :return: the new logger
        :rtype: KeyValueSection
        """
        return KeyValueSection(
            title=None, logger=self._logger,
            prefix=((prefix if (self._prefix is None) else
                     (self._prefix + logging.sanitize_name(prefix))) + "."),
            done=self.__done)


class TextSection(_Section):
    """A logger for raw, unprocessed text."""

    def __init__(self, title: str, logger: Logger) -> None:
        """
        Internal constructor, do not call it directly.

        :param str title: the title
        :param Logger logger: the logger
        """
        super().__init__(title, logger)

    def write(self, string: str) -> None:
        """
        Write a raw string to the logger.

        :param str string: the string to be written
        """
        # noinspection PyProtectedMember
        self._logger._write(string)
