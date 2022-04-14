"""Classes for printing tables in a text format."""

from contextlib import AbstractContextManager
from io import TextIOBase
from typing import Final, Optional, Iterable, Union, Callable, List

from moptipy.utils.path import Path
from moptipy.utils.types import type_error


class FormattedStr(str):
    """
    A subclass of `str` capable of holding formatting information.

    This is a very clunky method to pass either normal strings (instances
    of `str`) or formatted strings (instances of :class:`FormattedStr`) to
    the method :meth:`Row.cell` for rendering. The idea is that you can
    construct a list of strings in memory and attach formatting to them
    as needed and then render all of them via the same outlet.
    """

    #: should this string be formatted in bold face?
    bold: bool
    #: should this string be formatted in italic face?
    italic: bool
    #: should this string be formatted in code face?
    code: bool

    def __new__(cls, value, bold: bool = False, italic: bool = False,
                code: bool = False):
        """
        Construct the object.

        :param value: the string value
        """
        if not isinstance(bold, bool):
            raise type_error(bold, "bold", bool)
        if not isinstance(italic, bool):
            raise type_error(italic, "italic", bool)
        if not isinstance(code, bool):
            raise type_error(code, "code", bool)
        if bold or italic or code:
            ret = super(FormattedStr, cls).__new__(cls, value)
            ret.bold = bold
            ret.italic = italic
            ret.code = code
            return ret
        return value


class TableDriver:
    """
    A base class for table drivers.

    Table drivers allow us to render the structured text written to an
    instance of :class:`~moptipy.utils.table.Table` to a stream in a
    table format.
    """

    def begin_table(self, stream: TextIOBase, cols: str) -> None:
        """
        Write the beginning of the table.

        :param stream: the stream to write to
        :param cols: the column definition
        """

    def end_table(self, stream: TextIOBase, cols: str) -> None:
        """
        Write the ending of the table.

        :param stream: the stream to write to
        :param cols: the column definition
        """

    def begin_header(self, stream: TextIOBase, cols: str) -> None:
        """
        Begin the header row of the table.

        :param stream: the stream to write to
        :param cols: the column definition
        """

    def end_header(self, stream: TextIOBase, cols: str) -> None:
        """
        End the header row of the table.

        :param stream: the stream to write to
        :param cols: the column definition
        """

    def begin_section(self, stream: TextIOBase, cols: str,
                      section_index: int) -> None:
        """
        Begin a new section of the table.

        :param stream: the stream to write to
        :param cols: the column definition
        :param section_index: the index of the section, `0` for the first
            section
        """

    def end_section(self, stream: TextIOBase, cols: str,
                    section_index: int, had_header: bool, n_rows: int) -> None:
        """
        End a section of the table.

        :param stream: the stream to write to
        :param cols: the column definition
        :param section_index: the index of the section, `0` for the first
            section
        :param had_header: did the section have a header?
        :param n_rows: the number of rows that were written in the section
        """

    def begin_section_header(self, stream: TextIOBase, cols: str,
                             section_index: int) -> None:
        """
        Begin the header row of a section.

        :param stream: the stream to write to
        :param cols: the column definition
        :param section_index: the index of the section, `0` for the first
            section
        """

    def end_section_header(self, stream: TextIOBase, cols: str,
                           section_index: int) -> None:
        """
        End the header row of a section.

        :param stream: the stream to write to
        :param cols: the column definition
        :param section_index: the index of the section, `0` for the first
            section
        """

    def begin_row(self, stream: TextIOBase, cols: str,
                  section_index: int, row_index: int) -> None:
        """
        Begin a table header row, section row, or normal row in a section.

        :param stream: the stream to write to
        :param cols: the column definition
        :param section_index: the index of the current section, `-1` if we
            are in the table header
        :param row_index: the row index in the section: `0` for the first
            actual row, `-1` for a section header row, `-2` for a table
            header row
        """

    def end_row(self, stream: TextIOBase, cols: str,
                section_index: int, row_index: int) -> None:
        """
        End a table header row, section row, or normal row in a section.

        :param stream: the stream to write to
        :param cols: the column definition
        :param section_index: the index of the current section
        :param row_index: the row index in the section: `0` for the first
            actual row, `-1` for a section header row, `-2` for a table
            header row
        """

    def begin_cell(self, stream: TextIOBase, cols: str, section_index: int,
                   row_index: int, col_index: int) -> None:
        """
        Begin a header cell, section header cell, or normal cell.

        :param stream: the stream to write to
        :param cols: the column definitions
        :param section_index: the index of the current section, `-1` if this
            is a table header cell
        :param row_index: the row index in the section: `0` for the first
            actual row, `-1` for a section header row, `-2` for a table
            header row
        :param col_index: the column index, `0` for the first column
        """

    def text(self, stream: TextIOBase, text: str, bold: bool, italic: bool,
             code: bool) -> None:
        """
        Write the chunk of text of cell.

        :param stream: the stream to write to
        :param text: the text to write
        :param bold: is the text in bold face?
        :param italic: is the text in italic face?
        :param code: is the text in code face?
        """

    def end_cell(self, stream: TextIOBase, cols: str, section_index: int,
                 row_index: int, col_index: int) -> None:
        """
        End a header cell, section header cell, or normal cell.

        :param stream: the stream to write to
        :param cols: the column definitions
        :param section_index: the index of the current section, `-1` if this
            is a table header cell
        :param row_index: the row index in the section: `0` for the first
            actual row, `-1` for a section header row, `-2` for a table
            header row
        :param col_index: the column index, `0` for the first column
        """

    def exponent_renderer(self, e: str) -> str:
        """
        Render a numerical exponent.

        This function is for use in conjunction with
            :func:`moptipy.utils.strings.numbers_to_strings`.

        :param e: the exponent
        :returns: a rendered string
        """

    def filename(self,
                 file_name: str = "table",
                 dir_name: str = ".") -> Path:
        """
        Get the right filename for this table driver.

        :param file_name: the base file name
        :param dir_name: the base directory
        :returns: the path to the file to generate
        """
        if not isinstance(dir_name, str):
            raise type_error(dir_name, "dir_name", str)
        if len(dir_name) <= 0:
            raise ValueError(f"invalid dir_name: '{dir_name}'.")
        if not isinstance(file_name, str):
            raise type_error(file_name, "file_name", str)
        if len(file_name) <= 0:
            raise ValueError(f"invalid file_name: '{file_name}'.")
        out_dir = Path.directory(dir_name)
        suffix = self.__str__()
        if not isinstance(suffix, str):
            raise type_error(suffix, "result of str(table driver)", str)
        if len(suffix) <= 0:
            raise ValueError(f"invalid driver suffix: '{suffix}'")
        file: Final[Path] = out_dir.resolve_inside(f"{file_name}.{suffix}")
        file.ensure_file_exists()
        return file


class Markdown(TableDriver):
    r"""
    The markdown table driver.

    >>> from io import StringIO
    >>> s = StringIO()
    >>> md = Markdown()
    >>> print(str(md))
    md
    >>> with Table(s, "lrc", md) as t:
    ...     with t.header() as h:
    ...         h.cell(FormattedStr("1", bold=True))
    ...         h.cell(FormattedStr("2", code=True))
    ...         h.cell(FormattedStr("3", italic=True))
    ...     with t.section() as g:
    ...         with g.row() as r:
    ...             r.cell("a")
    ...             r.cell("b")
    ...             r.cell("c")
    ...         with g.row() as r:
    ...             r.cell("d")
    ...             r.cell("e")
    ...             r.cell("f")
    ...     print(f"'{s.getvalue()}'")
    '|**1**|`2`|*3*|
    |:--|--:|:-:|
    |a|b|c|
    |d|e|f|
    '
    """

    def end_header(self, stream: TextIOBase, cols: str) -> None:
        """End the header of a markdown table."""
        for c in cols:
            if c == "l":
                stream.write("|:--")
            elif c == "c":
                stream.write("|:-:")
            elif c == "r":
                stream.write("|--:")
            else:
                raise ValueError(f"Invalid col '{c}' in '{cols}'.")
        stream.write("|\n")

    def end_row(self, stream: TextIOBase, cols: str,
                section_index: int, row_index: int) -> None:
        """End a row in a Markdown table."""
        stream.write("|\n")

    def begin_cell(self, stream: TextIOBase, cols: str, section_index: int,
                   row_index: int, col_index: int) -> None:
        """Begin a Markdown table cell."""
        stream.write("|")

    def text(self, stream: TextIOBase, text: str, bold: bool, italic: bool,
             code: bool) -> None:
        """Print a table cell text string."""
        if len(text) <= 0:
            return
        if bold:
            stream.write("**")
        if italic:
            stream.write("*")
        if code:
            stream.write("`")
        stream.write(text)
        if code:
            stream.write("`")
        if italic:
            stream.write("*")
        if bold:
            stream.write("**")

    def __str__(self):
        """
        Get the file suffix.

        :returns: the file suffix
        :retval 'md': always
        """
        return "md"

    def exponent_renderer(self, e: str) -> str:
        """
        Render the numerical exponent in markdown.

        :param e: the exponent
        :returns: the rendered exponent
        :retval: `*10^{e}^`
        """
        return f"*10^{e}^"


class Table(AbstractContextManager):
    """
    The table context.

    This class provides a simple and hierarchically structured way to write
    tables in different formats. It only supports the most rudimentary
    formatting and nothing fancy (such as references, etc.). However, it may
    be totally enough to quickly produce tables with results of experiments.

    Every table must have a table header (see :meth:`header`).
    Every table then consists of a sequence of one or multiple sections
    (see :meth:`section` and :class:`Section`).
    Each table section itself may or may not have a header
    (see :meth:`Section.header`) and must have at least one row (see
    :meth:`Section.row` and :class:`Row`).
    Each row must have the exact right number of cells (see :meth:`Row.cell`).
    """

    def __init__(self, stream: TextIOBase, cols: str,
                 driver: TableDriver):
        """
        Initialize the table context.

        :param stream: the stream to which all output is written
        :param cols: the columns of the table
        :param driver: the table driver
        """
        super().__init__()
        if not isinstance(stream, TextIOBase):
            raise type_error(stream, "stream", TextIOBase)
        if not isinstance(cols, str):
            raise type_error(cols, "cols", str)
        if not isinstance(driver, TableDriver):
            raise type_error(driver, "driver", TableDriver)

        cols = cols.strip()
        if len(cols) <= 0:
            raise ValueError(
                "cols must not be empty to just composed of white space, "
                f"but is '{cols}'.")
        for c in cols:
            if c not in ("c", "l", "r"):
                raise ValueError("each col must be c, l, or r, but "
                                 f"encountered {c} in {cols}.")
        #: the internal stream
        self.__stream: TextIOBase = stream
        #: the internal column definition
        self.__cols: Final[str] = cols
        #: the internal table driver
        self.__driver: Final[TableDriver] = driver
        #: the header state: 0=no header, 1=in header, 2=after header
        self.__header_state: int = 0
        #: the section index
        self.__section_index: int = 0
        #: the row index
        self.__row_index: int = 0
        #: the column index
        self.__col_index: int = 0
        #: the section state: 0 outside of section, 1 inside of section,
        #: 2 after section
        self.__section_state: int = 0
        #: the section header state: 0=no header, 1=in header, 2=after header
        self.__section_header_state: int = 0
        #: the row state: 0=before row, 1=in row, 2=after row
        self.__row_state: int = 0

    def _begin_section(self):
        """Start a section."""
        if self.__stream is None:
            raise ValueError("table already closed, cannot start section.")
        if self.__header_state <= 0:
            raise ValueError("cannot start section before table header.")
        if self.__header_state <= 1:
            raise ValueError("cannot start section inside table header.")
        if self.__section_state == 1:
            raise ValueError("cannot start section inside section.")
        if self.__section_header_state == 1:
            raise ValueError("cannot start section inside section header.")
        if self.__section_header_state >= 1:
            raise ValueError(
                "cannot start section inside section after header.")
        if self.__row_state == 1:
            raise ValueError("cannot start section inside row.")
        self.__driver.begin_section(self.__stream, self.__cols,
                                    self.__section_index)
        self.__row_index = 0
        self.__section_state = 1
        self.__row_state = 0

    def _end_section(self):
        """End a section."""
        if self.__stream is None:
            raise ValueError("table already closed, cannot end section.")
        if self.__header_state <= 0:
            raise ValueError("cannot end section before table header.")
        if self.__header_state <= 1:
            raise ValueError("cannot end section inside table header.")
        if self.__section_state <= 0:
            raise ValueError("cannot start section outside section.")
        if self.__section_state >= 2:
            raise ValueError("cannot end section after section.")
        if self.__section_header_state == 1:
            raise ValueError("cannot end section inside section header.")
        if self.__row_index <= 0:
            raise ValueError(
                "cannot end section before writing any row.")
        if self.__row_state <= 0:
            raise ValueError("cannot end a section before a row.")
        if self.__row_state <= 1:
            raise ValueError("cannot end section inside of row.")
        self.__driver.end_section(self.__stream, self.__cols,
                                  self.__section_index,
                                  self.__section_header_state == 2,
                                  self.__row_index)
        self.__row_index = 0
        self.__section_state = 2
        self.__section_header_state = 0
        self.__section_index += 1

    def _begin_row(self, header_mode: int):
        """
        Start a row.

        :param header_mode: the header mode: -2 table header, -1 section
            header, 0 normal row
        """
        if self.__stream is None:
            raise ValueError("table already closed, cannot start row.")
        if self.__header_state == 1:
            raise ValueError(
                "cannot start row inside table header.")
        if self.__section_header_state == 1:
            raise ValueError(
                "cannot start row inside section header.")
        if self.__section_state >= 2:
            raise ValueError(
                "cannot start row after section.")
        if self.__row_state == 1:
            raise ValueError("cannot start row inside row.")

        row_index: int
        sec_index: int

        if header_mode == -2:  # start table header
            if self.__header_state >= 2:
                raise ValueError(
                    "cannot start table header row after table header.")
            if self.__section_state == 1:
                raise ValueError(
                    "cannot start table header row inside section.")
            if self.__section_header_state >= 2:
                raise ValueError(
                    "cannot start table header row after section header.")
            if self.__row_state >= 2:
                raise ValueError("cannot start table header after row.")
            self.__header_state = 1
            self.__driver.begin_header(self.__stream, self.__cols)
            row_index = -2
            sec_index = -1

        elif header_mode == -1:  # start section header
            if self.__header_state < 2:
                raise ValueError(
                    "can only start section header row after table header.")
            if self.__section_header_state >= 2:
                raise ValueError(
                    "cannot start section header row after section header.")
            if self.__section_state <= 0:
                raise ValueError(
                    "cannot start section header row outside of section.")
            if self.__row_index > 0:
                raise ValueError("cannot start section header row "
                                 f"after {self.__row_index} rows.")
            if self.__row_state >= 2:
                raise ValueError("cannot start section header after row.")
            self.__section_header_state = 1
            sec_index = self.__section_index
            self.__driver.begin_section_header(self.__stream, self.__cols,
                                               sec_index)
            row_index = -1

        elif header_mode == 0:
            if self.__header_state < 2:
                raise ValueError("can only start row after table header.")
            if self.__section_state <= 0:
                raise ValueError("cannot start row outside of section.")
            sec_index = self.__section_index
            row_index = self.__row_index
        else:
            raise ValueError(f"invalid header mode {header_mode}.")

        self.__col_index = 0
        self.__row_state = 1
        self.__driver.begin_row(self.__stream, self.__cols,
                                sec_index, row_index)

    def _end_row(self, header_mode: int):
        """
        End a row.

        :param header_mode: the header mode: -2 table header, -1 section
            header, 0 normal row
        """
        if self.__stream is None:
            raise ValueError("table already closed, cannot start row.")
        if self.__header_state == 0:
            raise ValueError(
                "cannot end row before table header.")
        if self.__section_state >= 2:
            raise ValueError("cannot end row after section.")
        if self.__row_state <= 0:
            raise ValueError("cannot end row before row begins.")
        if self.__row_state >= 2:
            raise ValueError("cannot end row after row has already ended.")
        if self.__col_index != len(self.__cols):
            raise ValueError(
                f"cannot end row after {self.__col_index} columns for table "
                f"with column definition {self.__cols}.")

        row_index: int
        sec_index: int

        if header_mode == -2:  # end table header
            if self.__header_state >= 2:
                raise ValueError(
                    "cannot end table header row after table header.")
            if self.__section_state == 1:
                raise ValueError(
                    "cannot end table header row inside section.")
            if self.__section_header_state == 1:
                raise ValueError(
                    "cannot end table header row inside section header.")
            if self.__section_header_state >= 2:
                raise ValueError(
                    "cannot end table header row after section header.")
            self.__header_state = 2
            row_index = -2
            sec_index = -1

        elif header_mode == -1:  # end section header
            if self.__header_state < 2:
                raise ValueError(
                    "can only end section header row after table header.")
            if self.__section_header_state <= 0:
                raise ValueError(
                    "cannot end section header row before section header.")
            if self.__section_header_state >= 2:
                raise ValueError(
                    "cannot end section header row after section header.")
            if self.__section_state <= 0:
                raise ValueError(
                    "cannot start section header row outside of section.")
            if self.__row_index > 0:
                raise ValueError("cannot end section header row "
                                 f"after {self.__row_index} rows.")
            self.__section_header_state = 2
            sec_index = self.__section_index
            row_index = -1

        elif header_mode == 0:
            if self.__header_state < 2:
                raise ValueError("can only end row after table header.")
            if self.__section_state <= 0:
                raise ValueError("cannot end row outside of section.")
            sec_index = self.__section_index
            row_index = self.__row_index
            self.__row_index += 1
        else:
            raise ValueError(f"invalid header mode {header_mode}.")

        self.__col_index = 0
        self.__row_state = 2
        self.__driver.end_row(self.__stream, self.__cols,
                              sec_index, row_index)
        if header_mode == -2:
            self.__driver.end_header(self.__stream, self.__cols)
        elif header_mode == -1:
            self.__driver.end_section_header(self.__stream, self.__cols,
                                             sec_index)

    def _cell(self, text: Optional[Union[str, Iterable[str]]]):
        """
        Render a cell.

        :param text: the text to write
        """
        if self.__stream is None:
            raise ValueError("table already closed, cannot start row.")
        if self.__header_state == 0:
            raise ValueError(
                "cannot have a cell before the table header starts.")
        if self.__section_state >= 2:
            raise ValueError(
                "cannot have cell after section end.")
        col_index: Final[int] = self.__col_index
        if col_index >= len(self.__cols):
            raise ValueError(
                f"cannot add cell after {col_index} columns for table "
                f"with column definition {self.__cols}.")
        if self.__row_state <= 0:
            raise ValueError("cannot begin cell before beginning a row.")
        if self.__row_state >= 2:
            raise ValueError("cannot begin cell after end of a row.")

        row_index: int
        section_index: int = self.__section_index
        if self.__header_state >= 2:
            if self.__section_state <= 0:
                raise ValueError(
                    "cannot begin cell after header outside of section.")
            row_index = -2
            section_index = -1
        elif self.__section_header_state == 1:
            row_index = -1
        else:
            row_index = self.__row_index
        self.__col_index = col_index + 1

        self.__driver.begin_cell(self.__stream, self.__cols, section_index,
                                 row_index, col_index)

        def __printit(st, strm: TextIOBase = self.__stream,
                      wrt: Callable[[TextIOBase, str, bool,
                                     bool, bool], None] = self.__driver.text) \
                -> None:
            if isinstance(st, str):
                if isinstance(st, FormattedStr):
                    wrt(strm, st, st.bold, st.italic, st.code)
                else:
                    wrt(strm, st, False, False, False)
            elif isinstance(st, Iterable):
                for ss in st:
                    __printit(ss)
            else:
                raise type_error(st, "text", (Iterable, str))
        __printit(text)
        self.__driver.end_cell(self.__stream, self.__cols, section_index,
                               row_index, col_index)

    def header(self) -> 'Row':
        """
        Construct the header of the table.

        :returns: a new managed header row
        """
        return Row(self, -2)

    def header_row(self, cells: Iterable[str]) -> None:
        """Print the header row with a single call."""
        if not isinstance(cells, Iterable):
            raise type_error(cells, "cells", Iterable)
        with self.header() as row:
            for cell in cells:
                row.cell(cell)

    def section(self) -> 'Section':
        """
        Create a new section of rows.

        :returns: a new managed row section
        """
        return Section(self)

    def section_cols(self, cols: List[List[Optional[str]]],
                     header_row: Optional[Iterable[Optional[str]]] = None):
        """
        Print a section columns-by-column.

        :param cols: an array which contains one list per column of the table.
        :param header_row: an optional header row
        """
        if not isinstance(cols, list):
            raise type_error(cols, "cols", list)
        if len(cols) != len(self.__cols):
            raise ValueError(
                f"expected {len(self.__cols)} columns ({self.__cols}), "
                f"but cols has length {len(cols)}.")
        max_rows = max(len(col) for col in cols)
        if max_rows <= 0:
            raise ValueError("There are no rows in the cols array?")

        with self.section() as sec:
            if header_row is not None:
                if not isinstance(header_row, Iterable):
                    raise type_error(header_row, "section header_row",
                                     Iterable)
                with sec.header() as head:
                    for cell in header_row:
                        head.cell(cell)

            for rowi in range(max_rows):
                with sec.row() as row:
                    for col in cols:
                        row.cell(None if rowi >= len(col) else col[rowi])

    def __enter__(self):
        """
        Enter the table in a `with` statement.

        :return: `self`
        """
        if self.__stream is None:
            raise ValueError("Table writing already finished!")
        self.__driver.begin_table(self.__stream, self.__cols)
        return self

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        """
        Close the table after leaving the `with` statement.

        :param exception_type: ignored
        :param exception_value: ignored
        :param traceback: ignored
        """
        if not (self.__stream is None):
            self.__driver.end_table(self.__stream, self.__cols)
            self.__stream.close()
            self.__stream = None
        if self.__section_state <= 0:
            raise ValueError("cannot end table before any section")
        if self.__section_state <= 1:
            raise ValueError("cannot end table inside a section")
        if self.__header_state <= 0:
            raise ValueError("cannot end table before table header")
        if self.__header_state <= 1:
            raise ValueError("cannot end table inside table header")
        if self.__section_header_state == 1:
            raise ValueError("cannot end table inside section header")


class Section(AbstractContextManager):
    """A table section is a group of rows, potentially with a header."""

    def __init__(self, owner: Table):
        """
        Initialize the row section.

        :param owner: the owning table
        """
        if not isinstance(owner, Table):
            raise type_error(owner, "owner", Table)
        #: the owner
        self.__owner: Final[Table] = owner

    def __enter__(self):
        """
        Enter the row section in a `with` statement.

        :return: `self`
        """
        # noinspection PyProtectedMember
        self.__owner._begin_section()
        return self

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        """
        Close the row section after leaving the `with` statement.

        :param exception_type: ignored
        :param exception_value: ignored
        :param traceback: ignored
        """
        # noinspection PyProtectedMember
        self.__owner._end_section()

    def header(self) -> 'Row':
        """
        Print the section header.

        :return: the header row
        """
        return Row(self.__owner, -1)

    def row(self) -> 'Row':
        """
        Create a row.

        :return: the new row
        """
        return Row(self.__owner, 0)


class Row(AbstractContextManager):
    """A row class."""

    def __init__(self, owner: Table, header_mode: int):
        """
        Initialize the row.

        :param owner: the owning table
        :param header_mode: the header mode
        """
        if not isinstance(owner, Table):
            raise type_error(owner, "owner", Table)
        if not isinstance(header_mode, int):
            raise type_error(header_mode, "header_mode", int)
        if not -2 <= header_mode <= 0:
            raise ValueError(f"Invalid header mode {header_mode}.")
        #: the owner
        self.__owner: Final[Table] = owner
        #: the header mode
        self.__header_mode: Final[int] = header_mode

    def cell(self, text: Optional[Union[str, Iterable[str]]] = None) -> None:
        """
        Render the text of a cell.

        As parameter `text`, you can provide either a string or a sequence of
        strings. You can also provide an instance of :class:`FormattedStr` or
        a sequence thereof. This allows you to render formatted text in a
        natural fashion.

        :param text: the text to write
        """
        # noinspection PyProtectedMember
        self.__owner._cell(text)

    def __enter__(self):
        """
        Enter the row in a `with` statement.

        :return: `self`
        """
        # noinspection PyProtectedMember
        self.__owner._begin_row(self.__header_mode)
        return self

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        """
        Close the row after leaving the `with` statement.

        :param exception_type: ignored
        :param exception_value: ignored
        :param traceback: ignored
        """
        # noinspection PyProtectedMember
        self.__owner._end_row(self.__header_mode)
