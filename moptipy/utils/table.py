"""Classes for printing tables in a text format."""

from contextlib import AbstractContextManager
from io import TextIOBase
from typing import Final


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
        Write the beginning of the header.

        :param stream: the stream to write to
        :param cols: the column definition
        """

    def end_header(self, stream: TextIOBase, cols: str) -> None:
        """
        Write the ending of the header.

        :param stream: the stream to write to
        :param cols: the column definition
        """

    def begin_row_group(self, stream: TextIOBase, cols: str, nrows: int,
                        ngroups: int) -> None:
        """
        Write the beginning of a new row group begins.

        :param stream: the stream to write to
        :param cols: the column definition
        :param nrows: the number of rows that have started so far
        :param ngroups: the number of row groups that have started so far
        """

    def end_row_group(self, stream: TextIOBase, cols: str, nrows: int,
                      ngroups: int, nrow_in_group: int) -> None:
        """
        Write the ending of a group of rows.

        :param stream: the stream to write to
        :param cols: the column definition
        :param nrows: the number of rows that have started so far
        :param ngroups: the number of row groups that have started so far
        :param nrow_in_group: the number of rows that have started so far in
            the current row group
        """

    def begin_row(self, stream: TextIOBase, cols: str, nrows: int,
                  ngroups: int, nrow_in_group: int, is_header: bool) -> None:
        """
        Write the beginning of a row.

        :param stream: the stream to write to
        :param cols: the column definition
        :param nrows: the number of rows that have started so far
        :param ngroups: the number of row groups that have started so far
        :param nrow_in_group: the number of rows that have started so far in
            the current row group
        :param is_header: is this a header row?
        """

    def end_row(self, stream: TextIOBase, cols: str, nrows: int,
                ngroups: int, nrow_in_group: int, is_header: bool) -> None:
        """
        Write the ending of a row.

        :param stream: the stream to write to
        :param cols: the column definition
        :param nrows: the number of rows that have started so far
        :param ngroups: the number of row groups that have started so far
        :param nrow_in_group: the number of rows that have started so far in
            the current row group
        :param is_header: is this a header row?
        """

    def cell(self, stream: TextIOBase, text: str, cols: str,
             colidx: int, bold: bool, italic: bool, code: bool,
             nrows: int, ngroups: int, nrow_in_group: int,
             is_header: bool) -> None:
        """
        Write the text of a cell.

        :param stream: the stream to write to
        :param text: the text to write
        :param cols: the column definitions
        :param colidx: the index of this cell
        :param bold: should the text be in bold face?
        :param italic: should the text be in italic face?
        :param code: should the text be in code face?
        :param nrows: the number of rows that have started so far
        :param ngroups: the number of row groups that have started so far
        :param nrow_in_group: the number of rows that have started so far in
            the current row group
        :param is_header: is this a header cell?
        """


class Table(AbstractContextManager):
    """
    The table context.

    This class provides a simple and hierarchically structured way to write
    tables in different formats. It only supports the most rudimentary
    formatting and nothing fancy (such as references, etc.). However, it may
    be totally enough to quickly produce tables with results of experiments.
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
            raise TypeError(
                f"stream should be TextIOBase, but is {type(stream)}.")
        if not isinstance(cols, str):
            raise TypeError(f"cols should be str, but is {type(cols)}.")
        if not isinstance(driver, TableDriver):
            raise TypeError(
                f"driver must be TableDriver, but is {type(driver)}.")

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
        #: the number of rows
        self.__nrows: int = 0
        #: the number of row groups
        self.__ngroups: int = 0
        #: the number of rows in the current row group
        self.__nrows_in_group: int = 0
        #: are we in a row group?
        self.__in_row_group: bool = False
        #: are we in a row?
        self.__in_row: bool = False
        #: the cell index inside the row
        self.__cell: int = 0
        #: 0=no header yet, 1=in header, 2=after header
        self.__header_mode: int = 0

    def _start_group(self) -> None:
        """Start a row group."""
        if self.__in_row_group:
            raise ValueError("Cannot start row group inside row!")
        if self.__in_row:
            raise ValueError("Cannot start row group inside row!")
        if self.__header_mode <= 0:
            raise ValueError("Cannot start row group before header.")
        if self.__header_mode <= 1:
            raise ValueError("Cannot start row group while in header.")
        self.__in_row_group = True
        self.__ngroups += 1
        self.__nrows_in_group = 0
        self.__cell = 0
        self.__driver.begin_row_group(self.__stream, self.__cols,
                                      self.__nrows, self.__ngroups)

    def _end_group(self) -> None:
        """End a row group."""
        if self.__header_mode <= 0:
            raise ValueError("Cannot end row group before header.")
        if self.__header_mode <= 1:
            raise ValueError("Cannot end row group while in header.")
        if self.__in_row:
            raise ValueError("Cannot end row group inside row!")
        if not self.__in_row_group:
            raise ValueError("Cannot end row group if not inside row group!")
        if self.__nrows_in_group <= 0:
            raise ValueError("No row in group?")
        self.__in_row_group = False
        self.__cell = 0
        self.__driver.end_row_group(self.__stream, self.__cols,
                                    self.__nrows, self.__ngroups,
                                    self.__nrows_in_group)
        self.__nrows_in_group = 0

    def _start_row(self, is_header: bool) -> None:
        """
        Start a row.

        :param is_header: are we in the header?
        """
        if is_header:
            if self.__header_mode >= 2:
                raise ValueError("Cannot start header row after header.")
            if self.__header_mode >= 1:
                raise ValueError("Cannot start header row inside header row.")
            self.__header_mode = 1
        else:
            if self.__header_mode < 2:
                raise ValueError("Can end row only after header.")
            if not self.__in_row_group:
                raise ValueError("Cannot start row outside of row group!")
        if self.__in_row:
            raise ValueError("Cannot start row inside row!")
        self.__nrows_in_group += 1
        self.__nrows += 1
        self.__in_row = True
        self.__cell = 0
        if is_header:
            self.__driver.begin_header(self.__stream, self.__cols)
        self.__driver.begin_row(self.__stream, self.__cols, self.__nrows,
                                self.__ngroups, self.__nrows_in_group,
                                is_header)

    def _end_row(self, is_header: bool) -> None:
        """
        End a row.

        :param is_header: are we in the header?
        """
        if is_header:
            if self.__header_mode <= 0:
                raise ValueError("Cannot end header row before header.")
            if self.__header_mode >= 2:
                raise ValueError("Cannot end header row after header.")
            self.__header_mode = 2
        else:
            if self.__header_mode < 2:
                raise ValueError("Can end row only after header.")
            if not self.__in_row_group:
                raise ValueError("Cannot end row if not inside row group!")

        if not self.__in_row:
            raise ValueError("Cannot end row group outside row!")
        if self.__cell != len(self.__cols):
            raise ValueError(f"Only specified {self.__cell} cells in "
                             f"row of type '{self.__cols}'.")
        self.__in_row = False
        self.__cell = 0
        self.__driver.end_row(self.__stream, self.__cols, self.__nrows,
                              self.__ngroups, self.__nrows_in_group,
                              is_header)
        if is_header:
            self.__driver.end_header(self.__stream, self.__cols)

    def _cell(self, text: str, bold: bool, italic: bool, code: bool,
              is_header: bool) -> None:
        """
        Render a cell.

        :param text: the text to write
        :param bold: should the text be in bold face?
        :param italic: should the text be in italic face?
        :param code: should the text be in code face?
        :param is_header: are we in the header?
        """
        if not self.__in_row:
            raise ValueError(f"Can only start cell '{text}' inside of row.")
        if self.__header_mode <= 0:
            raise ValueError(f"Cannot have cell '{text}' before header.")
        if self.__header_mode >= 1:
            if not (is_header or self.__in_row_group):
                raise ValueError(
                    f"Can only start cell '{text}' inside of row group.")
        if (self.__header_mode == 1) != is_header:
            raise ValueError(f"expected {self.__header_mode} == 1 to "
                             f"be {is_header} for cell '{text}'.")
        c: Final[int] = self.__cell
        if c >= len(self.__cols):
            raise ValueError(f"Cannot put cell '{text}' at index "
                             f"{c + 1} for cols '{self.__cols}'.")
        self.__cell = c + 1
        self.__driver.cell(self.__stream, text.strip(), self.__cols, c, bold,
                           italic, code, self.__nrows, self.__ngroups,
                           self.__nrows_in_group, is_header)

    def header(self) -> 'Row':
        """
        Start the header row.

        :returns: the header row
        """
        return Row(self, True)

    def group(self) -> 'RowGroup':
        """
        Create a new group of rows.

        :returns: a new managed row group
        """
        return RowGroup(self)

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
        if self.__header_mode != 2:
            raise ValueError("Cannot exit table before closing header.")
        if self.__in_row:
            raise ValueError("Cannot exit table inside row!")
        if self.__in_row_group:
            raise ValueError("Cannot exit table inside row group!")


class RowGroup(AbstractContextManager):
    """A group of rows."""

    def __init__(self, owner: Table):
        """
        Initialize the row group.

        :param owner: the owning table
        """
        #: the owner
        self.__owner: Final[Table] = owner

    def __enter__(self):
        """
        Enter the row group in a `with` statement.

        :return: `self`
        """
        # noinspection PyProtectedMember
        self.__owner._start_group()
        return self

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        """
        Close the row group after leaving the `with` statement.

        :param exception_type: ignored
        :param exception_value: ignored
        :param traceback: ignored
        """
        # noinspection PyProtectedMember
        self.__owner._end_group()

    def row(self) -> 'Row':
        """
        Create a row.

        :return: the new row
        """
        return Row(self.__owner, False)


class Row(AbstractContextManager):
    """A row class."""

    def __init__(self, owner: Table, is_header: bool):
        """
        Initialize the row.

        :param owner: the owning table
        :param is_header: are we in the header
        """
        #: the owner
        self.__owner: Final[Table] = owner
        #: are we in the header?
        self.__is_header = is_header

    def cell(self, text: str = "", bold: bool = False,
             italic: bool = False, code: bool = False) -> None:
        """
        Render a cell.

        :param text: the text to write
        :param bold: should the text be in bold face?
        :param italic: should the text be in italic face?
        :param code: should the text be in code face?
        """
        # noinspection PyProtectedMember
        self.__owner._cell(text, bold, italic, code, self.__is_header)

    def __enter__(self):
        """
        Enter the row in a `with` statement.

        :return: `self`
        """
        # noinspection PyProtectedMember
        self.__owner._start_row(self.__is_header)
        return self

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        """
        Close the row after leaving the `with` statement.

        :param exception_type: ignored
        :param exception_value: ignored
        :param traceback: ignored
        """
        # noinspection PyProtectedMember
        self.__owner._end_row(self.__is_header)


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
    ...     with t.group() as g:
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

    def end_row(self, stream: TextIOBase, cols: str, nrows: int,
                ngroups: int, nrow_in_group: int, is_header: bool) -> None:
        """End a row in a Markdown table."""
        stream.write("|\n")

    def cell(self, stream: TextIOBase, text: str, cols: str,
             colidx: int, bold: bool, italic: bool, code: bool,
             nrows: int, ngroups: int, nrow_in_group: int,
             is_header: bool) -> None:
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
