"""Classes for printing tables in a text format."""

from contextlib import AbstractContextManager
from io import TextIOBase
from typing import Final, Optional

from moptipy.utils.types import type_error


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

    def cell(self, stream: TextIOBase, text: str, cols: str,
             section_index: int, row_index: int, col_index: int,
             bold: bool, italic: bool, code: bool) -> None:
        """
        Write the text of a header cell, section header cell, or normal cell.

        :param stream: the stream to write to
        :param text: the text to write
        :param cols: the column definitions
        :param section_index: the index of the current section, `-1` if this
            is a table header cell
        :param row_index: the row index in the section: `0` for the first
            actual row, `-1` for a section header row, `-2` for a table
            header row
        :param col_index: the column index, `0` for the first column
        :param bold: should the text be in bold face?
        :param italic: should the text be in italic face?
        :param code: should the text be in code face?
        """


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

    def _cell(self, text: Optional[str] = None, bold: bool = False,
              italic: bool = False, code: bool = False):
        """
        Render a cell.

        :param text: the text to write
        :param bold: should the text be in bold face?
        :param italic: should the text be in italic face?
        :param code: should the text be in code face?
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
        self.__driver.cell(self.__stream, text, self.__cols, section_index,
                           row_index, col_index, bold, italic, code)

    def header(self) -> 'Row':
        """
        Construct the header of the table.

        :returns: a new managed header row
        """
        return Row(self, -2)

    def section(self) -> 'Section':
        """
        Create a new section of rows.

        :returns: a new managed row section
        """
        return Section(self)

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

    def cell(self, text: Optional[str] = None, bold: bool = False,
             italic: bool = False, code: bool = False) -> None:
        """
        Render a cell.

        :param text: the text to write
        :param bold: should the text be in bold face?
        :param italic: should the text be in italic face?
        :param code: should the text be in code face?
        """
        # noinspection PyProtectedMember
        self.__owner._cell(text, bold, italic, code)

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


class Markdown(TableDriver):
    r"""
    The markdown table driver.

    >>> from io import StringIO
    >>> s = StringIO()
    >>> with Table(s, "lrc", Markdown()) as t:
    ...     with t.header() as h:
    ...         h.cell("1", bold=True)
    ...         h.cell("2", code=True)
    ...         h.cell("3", italic=True)
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

    def cell(self, stream: TextIOBase, text: str, cols: str,
             section_index: int, row_index: int, col_index: int,
             bold: bool, italic: bool, code: bool) -> None:
        """Write a Markdown table cell."""
        stream.write("|")
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
