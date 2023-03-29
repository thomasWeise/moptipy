"""
Classes for writing structured log files.

A :class:`~Logger` offers functionality to write structured, text-based log
files that can hold a variety of information. It is implemented in two
flavors, :class:`~FileLogger`, which writes data to a file, and
:class:`~InMemoryLogger`, which writes data to a buffer in memory (which is
mainly useful for testing).

A :class:`~Logger` can produce output in three formats:

- :meth:`~Logger.csv` creates a section of semicolon-separated-values data
  (:class:`~CsvLogSection`), which we call `csv` because it structured
  basically exactly as the well-known comma-separated-values data, just
  using semicolons.
- :meth:`~Logger.key_values` creates a key-values section
  (:class:`~KeyValueLogSection`) in YAML format. The specialty of this
  section is that it permits hierarchically structuring of data by spawning
  out sub-sections that are signified by a key prefix via
  :meth:`~KeyValueLogSection.scope`.
- :meth:`~Logger.text` creates a raw text section (:class:`~TextLogSection`),
  into which raw text can be written.

The beginning and ending of section named `XXX` are `BEGIN_XXX` and `END_XXX`.

This class is used by the :class:`~moptipy.api.execution.Execution` and
experiment-running facility (:func:`~moptipy.api.experiment.run_experiment`)
to produce log files complying with
https://thomasweise.github.io/moptipy/#log-files.
Such log files can be parsed via :mod:`~moptipy.evaluation.log_parser`.
"""

from contextlib import AbstractContextManager
from io import StringIO, TextIOBase
from math import isfinite
from os.path import realpath
from re import sub
from typing import Callable, Final, Iterable, cast

from moptipy.utils.cache import is_new
from moptipy.utils.path import Path
from moptipy.utils.strings import (
    PART_SEPARATOR,
    bool_to_str,
    float_to_str,
    sanitize_name,
)
from moptipy.utils.types import type_error

#: the separator used in CSV files to separate columns
CSV_SEPARATOR: Final[str] = ";"
#: the character indicating the begin of a comment
COMMENT_CHAR: Final[str] = "#"
#: the character separating a scope prefix in a key-value section
SCOPE_SEPARATOR: Final[str] = "."
#: the indicator of the start of a log section
SECTION_START: Final[str] = "BEGIN_"
#: the indicator of the end of a log section
SECTION_END: Final[str] = "END_"
#: the replacement for special characters
SPECIAL_CHAR_REPLACEMENT: Final[str] = PART_SEPARATOR
#: the YAML-conform separator between a key and a value
KEY_VALUE_SEPARATOR: Final[str] = ": "
#: the hexadecimal version of a value
KEY_HEX_VALUE: Final[str] = "(hex)"


class Logger(AbstractContextManager):
    """
    An abstract base class for logging data in a structured way.

    There are two implementations of this, :class:`InMemoryLogger`, which logs
    data in memory and is mainly there for testing and debugging, an
    :class:`FileLogger` which logs to a text file and is to be used in
    experiments with `moptipy`.
    """

    def __init__(self, stream: TextIOBase, name: str) -> None:
        """
        Create a new logger.

        :param stream: the stream to which we will log, will be closed when
            the logger is closed
        :param name: the name of the logger
        """
        if not isinstance(name, str):
            raise type_error(name, "name", str)
        if stream is None:
            raise ValueError("stream must be valid stream but is None.")
        if not isinstance(stream, TextIOBase):
            raise type_error(stream, "stream", TextIOBase)

        #: The internal stream
        self._stream: TextIOBase = stream
        self.__section: str | None = None
        self.__starts_new_line: bool = True
        self.__log_name: str = name
        self.__sections: Callable = is_new()
        self.__closer: str | None = None

    def __enter__(self):
        """
        Enter the logger in a `with` statement.

        :return: `self`
        """
        return self

    def _error(self, message: str) -> None:
        """
        Raise a :class:`ValueError` with context information.

        :param message: the message elements to merge
        :raises ValueError: an error with the message and some context
        information
        """
        raise ValueError(f"{message} in logger {self.__log_name!r}."
                         if self.__section is None else
                         f"{message} in section {self.__section!r} "
                         f"of logger {self.__log_name!r}.")

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        """
        Close the logger after leaving the `with` statement.

        :param exception_type: ignored
        :param exception_value: ignored
        :param traceback: ignored
        """
        if self.__section is not None:
            self._error("Cannot close logger, because section still open")
        if self._stream is not None:
            if not self.__starts_new_line:
                self._stream.write("\n")
            self._stream.close()
            self._stream = None

    def _open_section(self, title: str) -> None:
        """
        Open a new section.

        :param title: the section title
        """
        if self._stream is None:
            self._error(f"Cannot open section {title!r} "
                        "because logger already closed")

        if self.__section is not None:
            self._error(f"Cannot open section {title!r} because "
                        "another one is open")

        new_title = title.strip().upper()
        if new_title != title:
            self._error(f"Cannot open section {title!r} because "
                        "title is invalid")

        if not self.__sections(title):
            self._error(f"Section {title!r} already done")

        self._stream.write(f"{SECTION_START}{title}\n")
        self.__closer = f"{SECTION_END}{title}\n"
        self.__starts_new_line = True
        self.__section = title

    def _close_section(self, title: str) -> None:
        """
        Close a section.

        :param title: the section title
        """
        if (self.__section is None) or (self.__section != title):
            self._error(f"Cannot open section {title!r} since it is not open")
        printer = self.__closer
        if not self.__starts_new_line:
            printer = "\n" + printer

        self._stream.write(printer)
        self.__closer = None
        self.__starts_new_line = True
        self.__section = None

    def _comment(self, comment: str) -> None:
        """
        Write a comment.

        :param comment: the comment
        """
        if self.__section is None:
            self._error("Cannot write if not inside section")
        if len(comment) <= 0:
            return
        comment = sub(r"\s+", " ", comment.strip())
        self._stream.write(f"{COMMENT_CHAR} {comment}\n")
        self.__starts_new_line = True

    def _write(self, text: str) -> None:
        """
        Write a string.

        :param text: the text to write
        """
        if self.__section is None:
            self._error("Cannot write if not inside section")

        if len(text) <= 0:
            return

        if self.__closer in text:
            self._error(f"String {self.__closer!r} "
                        "must not be contained in output")

        text = text.replace("#", "")  # omit all # characters
        self._stream.write(text)
        self.__starts_new_line = text.endswith("\n")

    def key_values(self, title: str) -> "KeyValueLogSection":
        r"""
        Create a log section for key-value pairs.

        The contents of such a section will be valid YAML mappings, i.w.,
        conform to
        https://yaml.org/spec/1.2/spec.html#mapping.
        This means they can be parsed with a YAML parser (after removing the
        section start and end marker, of course).

        :param title: the title of the new section
        :return: the new logger

        >>> from moptipy.utils.temp import TempFile
        >>> with TempFile.create() as t:
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
        return KeyValueLogSection(title=title, logger=self, prefix="",
                                  done=None)

    def csv(self, title: str, header: list[str]) -> "CsvLogSection":
        """
        Create a log section for CSV data with `;` as column separator.

        The first line will be the headline with the column names.

        :param title: the title of the new section
        :param header: the list of column titles
        :return: the new logger

        >>> from moptipy.utils.logger import FileLogger
        >>> from moptipy.utils.temp import TempFile
        >>> with TempFile.create() as t:
        ...     with FileLogger(str(t)) as l:
        ...         with l.csv("A", ["x", "y"]) as csv:
        ...             csv.row([1,2])
        ...             csv.row([3,4])
        ...             csv.row([4, 12])
        ...     text = open(str(t), "r").read().splitlines()
        ...     print(text)
        ['BEGIN_A', 'x;y', '1;2', '3;4', '4;12', 'END_A']
        >>> import csv
        >>> for r in csv.reader(text[1:5], delimiter=";"):
        ...     print(r)
        ['x', 'y']
        ['1', '2']
        ['3', '4']
        ['4', '12']
        """
        return CsvLogSection(title=title, logger=self, header=header)

    def text(self, title: str) -> "TextLogSection":
        r"""
        Create a log section for unstructured text.

        :param title: the title of the new section
        :return: the new logger

        >>> from moptipy.utils.logger import InMemoryLogger
        >>> with InMemoryLogger() as l:
        ...     with l.text("C") as tx:
        ...         tx.write("aaaaaa")
        ...         tx.write("bbbbb")
        ...         tx.write("\n")
        ...         tx.write("ccccc")
        ...     print(l.get_log())
        ['BEGIN_C', 'aaaaaabbbbb', 'ccccc', 'END_C']
        """
        return TextLogSection(title=title, logger=self)


class FileLogger(Logger):
    """A logger logging to a file."""

    def __init__(self, path: str) -> None:
        """
        Initialize the logger.

        :param path: the path to the file to open
        """
        if not isinstance(path, str):
            raise type_error(path, "path", str)
        name = path
        path = realpath(path)
        super().__init__(stream=Path.path(path).open_for_write(),
                         name=name)


class InMemoryLogger(Logger):
    """A logger logging to a string in memory."""

    def __init__(self) -> None:
        """Initialize the logger."""
        super().__init__(stream=StringIO(),
                         name="in-memory-logger")

    def get_log(self) -> list[str]:
        """
        Obtain all the lines logged to this logger.

        :return: a list of strings with the logged lines
        """
        return cast(StringIO, self._stream).getvalue().splitlines()


class LogSection(AbstractContextManager):
    """An internal base class for logger sections."""

    def __init__(self, title: str | None, logger: Logger) -> None:
        """
        Perform internal construction. Do not call directly.

        :param title: the section title
        :param logger: the logger
        """
        self._logger: Logger = logger
        self._title: str | None = title
        if title is not None:
            # noinspection PyProtectedMember
            logger._open_section(title)

    def __enter__(self):
        """
        Enter the context: needed for the `with` statement.

        :return: `self`
        """
        return self

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        """
        Exit the `with` statement.

        :param exception_type: ignored
        :param exception_value: ignored
        :param traceback: ignored
        :return: ignored
        """
        if self._title is not None:
            # noinspection PyProtectedMember
            self._logger._close_section(self._title)
            self._title = None
        self._logger = None

    def comment(self, comment: str) -> None:
        """
        Write a comment line.

        A comment starts with `#` and is followed by text.

        :param comment: the comment to write

        >>> from moptipy.utils.logger import InMemoryLogger
        >>> with InMemoryLogger() as l:
        ...     with l.text("A") as tx:
        ...         tx.write("aaaaaa")
        ...         tx.comment("hello")
        ...     print(l.get_log())
        ['BEGIN_A', 'aaaaaa# hello', 'END_A']
        """
        # noinspection PyProtectedMember
        self._logger._comment(comment)


class CsvLogSection(LogSection):
    """
    A logger that is designed to output CSV data.

    The coma-separated-values log is actually a semicolon-separated-values
    log. This form of logging is used to store progress information or
    time series data, as captured by the optimization
    :class:`~moptipy.api.process.Process` and activated by, e.g., the methods
    :meth:`~moptipy.api.execution.Execution.set_log_improvements` and
    :meth:`~moptipy.api.execution.Execution.set_log_all_fes`. It will look
    like this:

    >>> with InMemoryLogger() as logger:
    ...     with logger.csv("CSV", ["A", "B", "C"]) as csv:
    ...         csv.row((1, 2, 3))
    ...         csv.row([1.5, 2.0, 3.5])
    ...     print(logger.get_log())
    ['BEGIN_CSV', 'A;B;C', '1;2;3', '1.5;2;3.5', 'END_CSV']
    """

    def __init__(self, title: str, logger: Logger, header: list[str]) -> None:
        """
        Perform internal construction. Do not call directly.

        :param title: the title
        :param logger: the owning logger
        :param header: the header
        """
        super().__init__(title, logger)

        self.__header_len: Final[int] = len(header)
        if self.__header_len <= 0:
            # noinspection PyProtectedMember
            logger._error(f"Empty header {header} invalid for a CSV section")

        for c in header:
            if (not (isinstance(c, str))) or CSV_SEPARATOR in c:
                # noinspection PyProtectedMember
                logger._error(f"Invalid column {c}")

        # noinspection PyProtectedMember
        logger._write(CSV_SEPARATOR.join(
            [c.strip() for c in header]) + "\n")

    def row(self, row: tuple[int | float | bool, ...]
            | list[int | float | bool]) -> None:
        """
        Write a row of csv data.

        :param row: the row of data
        """
        if self.__header_len != len(row):
            # noinspection PyProtectedMember
            self._logger._error(
                f"Header of CSV section demands {self.__header_len} columns, "
                f"but row {row} has {len(row)}")

        # noinspection PyProtectedMember
        txt = [str(c) if isinstance(c, int)
               else bool_to_str(c) if isinstance(c, bool)  # type: ignore
               else (float_to_str(c) if isinstance(c, float) else
               cast(None, self._logger._error(
                   f"Invalid log value {c} in row {row}")))
               for c in row]

        # noinspection PyProtectedMember
        self._logger._write(f"{CSV_SEPARATOR.join(txt)}\n")


class KeyValueLogSection(LogSection):
    """
    A logger for key-value pairs.

    The key-values section `XXX` starts with the line `BEGIN_XXX` and ends
    with the line `END_XXX`. On every line in between, there is a key-value
    pair of the form `key: value`. Key-values sections support so-called
    scopes. Key-values pairs belong to a scope `Y` if the key starts with `Y.`
    followed by the actual key, e.g., `a.g: 5` denotes that the key `g` of
    scope `a` has value `5`. Such scopes can be arbitrarily nested: The
    key-value pair `x.y.z: 2` denotes a key `z` in the scope `y` nested within
    scope `x` having the value `2`. This system of nested scopes allows you
    to recursively invoke the method
    :meth:`~moptipy.api.component.Component.log_parameters_to` without
    worrying of key clashes. Just wrap the call to the `log_parameters_to`
    method of a sub-component into a unique scope. At the same time, this
    avoids the need of any more complex hierarchical data structures in our
    log files.

    >>> with InMemoryLogger() as logger:
    ...     with logger.key_values("A") as kv:
    ...         kv.key_value("x", 1)
    ...         with kv.scope("b") as sc1:
    ...             sc1.key_value("i", "j")
    ...             with sc1.scope("c") as sc2:
    ...                 sc2.key_value("l", 5)
    ...         kv.key_value("y", True)
    ...     print(logger.get_log())
    ['BEGIN_A', 'x: 1', 'b.i: j', 'b.c.l: 5', 'y: T', 'END_A']
    """

    def __init__(self, title: str | None,
                 logger: Logger, prefix: str, done) -> None:
        """
        Perform internal construction, do not call directly.

        :param title: the section title, or `None` for nested scopes.
        :param logger: the owning logger
        :param prefix: the prefix
        :param done: the set of already done keys and prefixes
        """
        if not isinstance(prefix, str):
            raise type_error(prefix, "prefix", str)
        super().__init__(title=title, logger=logger)
        self._prefix: Final[str] = prefix
        self.__done: Callable
        if done is None:
            self.__done = is_new()
            self.__done(prefix)
        else:
            self.__done = done
            if not done(prefix):
                # noinspection PyProtectedMember
                logger._error(f"Prefix {prefix!r} already done")

    def key_value(self, key: str, value,
                  also_hex: bool = False) -> None:
        """
        Write a key-value pair.

        Given key `A` and value `B`, the line `A: B` will be added to the log.
        If `value` (`B`) happens to be a floating point number, the value will
        also be stored in hexadecimal notation (:meth:`float.hex`).

        :param key: the key
        :param value: the value
        :param also_hex: also store the value as hexadecimal version
        """
        key = self._prefix + sanitize_name(key)
        if not self.__done(key):
            # noinspection PyProtectedMember
            self._logger._error(f"Key {key!r} already used")

        the_hex = None
        if isinstance(value, float):
            txt = float_to_str(value)
            if isfinite(value) and (also_hex
                                    or (("e" in txt) or ("." in txt))):
                the_hex = float.hex(value)
        elif isinstance(value, bool):
            txt = bool_to_str(value)
        else:
            txt = str(value)
            if also_hex and isinstance(value, int):
                the_hex = hex(value)

        txt = KEY_VALUE_SEPARATOR.join([key, txt])
        txt = f"{txt}\n"

        if the_hex:
            tmp = KEY_VALUE_SEPARATOR.join(
                [key + KEY_HEX_VALUE, the_hex])
            txt = f"{txt}{tmp}\n"

        # noinspection PyProtectedMember
        self._logger._write(txt)

    def scope(self, prefix: str) -> "KeyValueLogSection":
        """
        Create a new scope for key prefixes.

        :class:`KeyValueLogSection` only allows you to store flat key-value
        pairs where each key must be unique. However, what do we do if we
        have two components of an algorithm that have parameters with the
        same name (key)?
        We can hierarchically nest :class:`KeyValueLogSection` sections via
        prefix scopes. The idea is as follows: If one component has
        sub-components, instead of invoking their
        :meth:`~moptipy.api.component.Component.log_parameters_to` methods
        directly, which could lead to key-clashes, it will create a
        :meth:`scope` for each one and then pass these scopes to their
        :meth:`~moptipy.api.component.Component.log_parameters_to`.
        Each scope basically appends a prefix and a "." to the keys.
        If the prefixes are unique, this ensures that all prefix+"."+keys are
        unique, too.

        >>> from moptipy.utils.logger import InMemoryLogger
        >>> with InMemoryLogger() as l:
        ...     with l.key_values("A") as kv:
        ...         kv.key_value("x", "y")
        ...         with kv.scope("b") as sc1:
        ...             sc1.key_value("x", "y")
        ...             with sc1.scope("c") as sc2:
        ...                 sc2.key_value("x", "y")
        ...         with kv.scope("d") as sc3:
        ...             sc3.key_value("x", "y")
        ...             with sc3.scope("c") as sc4:
        ...                sc4.key_value("x", "y")
        ...     print(l.get_log())
        ['BEGIN_A', 'x: y', 'b.x: y', 'b.c.x: y', 'd.x: y', 'd.c.x: y', \
'END_A']

        :param prefix: the key prefix
        :return: the new logger
        """
        return KeyValueLogSection(
            title=None, logger=self._logger,
            prefix=(prefix if (self._prefix is None) else
                    f"{self._prefix}{sanitize_name(prefix)}."),
            done=self.__done)


class TextLogSection(LogSection):
    """
    A logger for raw, unprocessed text.

    Such a log section is used to capture the raw contents of the solutions
    discovered by the optimization :class:`~moptipy.api.process.Process`. For
    this purpose, our system will use the method
    :meth:`~moptipy.api.space.Space.to_str` of the search and/or solution
    :class:`~moptipy.api.space.Space`.

    >>> with InMemoryLogger() as logger:
    ...     with logger.text("T") as txt:
    ...         txt.write("Hello World!")
    ...     print(logger.get_log())
    ['BEGIN_T', 'Hello World!', 'END_T']
    """

    def __init__(self, title: str, logger: Logger) -> None:
        """
        Perform internal construction. Do not call it directly.

        :param title: the title
        :param logger: the logger
        """
        super().__init__(title, logger)
        # noinspection PyProtectedMember
        self.write = self._logger._write  # type: ignore


def parse_key_values(lines: Iterable[str]) -> dict[str, str]:
    """
    Parse a :meth:`~moptipy.utils.logger.Logger.key_values` section's text.

    :param lines: the lines with the key-values pairs
    :return: the dictionary with the

    >>> from moptipy.utils.logger import InMemoryLogger
    >>> with InMemoryLogger() as l:
    ...     with l.key_values("B") as kv:
    ...         kv.key_value("a", "b")
    ...         with kv.scope("c") as kvc:
    ...             kvc.key_value("d", 12)
    ...             kvc.key_value("e", True)
    ...         kv.key_value("f", 3)
    ...     txt = l.get_log()
    >>> print(txt)
    ['BEGIN_B', 'a: b', 'c.d: 12', 'c.e: T', 'f: 3', 'END_B']
    >>> dic = parse_key_values(txt[1:5])
    >>> keys = list(dic.keys())
    >>> keys.sort()
    >>> print(keys)
    ['a', 'c.d', 'c.e', 'f']
    """
    if not isinstance(lines, Iterable):
        raise type_error(lines, "lines", Iterable)
    dct = {}
    for i, line in enumerate(lines):
        if not isinstance(line, str):
            raise type_error(line, f"lines[{i}]", str)
        splt = line.split(KEY_VALUE_SEPARATOR)
        if len(splt) != 2:
            raise ValueError(
                f"Two strings separated by {KEY_VALUE_SEPARATOR!r} "
                f"expected, but encountered {len(splt)} in {i}th "
                f"line {line!r}.")
        key = splt[0].strip()
        if len(key) <= 0:
            raise ValueError(
                f"Empty key encountered in {i}th line {line!r}.")
        value = splt[1].strip()
        if len(value) <= 0:
            raise ValueError(
                f"Empty value encountered in {i}th line {line!r}.")
        dct[key] = value

    return dct
