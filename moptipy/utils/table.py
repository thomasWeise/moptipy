"""Classes for printing tables in a text format."""

from contextlib import AbstractContextManager
from io import TextIOBase
from typing import Final, Optional, Iterable, Union, Callable, List

from moptipy.utils.text_format import TextFormatDriver, FormattedStr
from moptipy.utils.types import type_error


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
                 driver: TextFormatDriver):
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
        if not isinstance(driver, TextFormatDriver):
            raise type_error(driver, "driver", TextFormatDriver)

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
        self.__driver: Final[TextFormatDriver] = driver
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
        self.__driver.begin_table_section(self.__stream, self.__cols,
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
        self.__driver.end_table_section(
            self.__stream, self.__cols, self.__section_index,
            #  had_header is not used: self.__section_header_state == 2,
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
            self.__driver.begin_table_header(self.__stream, self.__cols)
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
            self.__driver.begin_table_section_header(
                self.__stream, self.__cols, sec_index)
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
        self.__driver.begin_table_row(
            self.__stream, self.__cols, sec_index, row_index)

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
        self.__driver.end_table_row(self.__stream, self.__cols,
                                    sec_index, row_index)
        if header_mode == -2:
            self.__driver.end_table_header(self.__stream, self.__cols)
        elif header_mode == -1:
            self.__driver.end_table_section_header(
                self.__stream, self.__cols, sec_index)

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
        if self.__header_state == 1:
            row_index = -2
            section_index = -1
        else:
            if self.__section_state <= 0:
                raise ValueError(
                    "cannot begin cell after header outside of section.")
            if self.__section_header_state == 1:
                row_index = -1
            else:
                row_index = self.__row_index
        self.__col_index = col_index + 1

        self.__driver.begin_table_cell(self.__stream, self.__cols,
                                       section_index, row_index, col_index)

        def __printit(st, strm: TextIOBase = self.__stream,
                      wrt: Callable[[TextIOBase, str, bool,
                                     bool, bool], None] = self.__driver.text) \
                -> None:
            if st is None:
                return
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
        self.__driver.end_table_cell(
            self.__stream, self.__cols, section_index, row_index, col_index)

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
        self.__driver.begin_table_body(self.__stream, self.__cols)
        return self

    def __exit__(self, exception_type, exception_value, traceback) -> bool:
        """
        Close the table after leaving the `with` statement.

        :param exception_type: ignored
        :param exception_value: ignored
        :param traceback: ignored
        :returns: `True` to suppress an exception, `False` to rethrow it
        """
        if not (self.__stream is None):
            self.__driver.end_table_body(self.__stream, self.__cols)
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
        return exception_type is None


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

    def __exit__(self, exception_type, exception_value, traceback) -> bool:
        """
        Close the row section after leaving the `with` statement.

        :param exception_type: ignored
        :param exception_value: ignored
        :param traceback: ignored
        :returns: `True` to suppress an exception, `False` to rethrow it
        """
        # noinspection PyProtectedMember
        self.__owner._end_section()
        return exception_type is None

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
        strings. You can also provide an instance of
        :class:`moptipy.utils.text_format.FormattedStr` or a sequence thereof.
        This allows you to render formatted text in a natural fashion.

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

    def __exit__(self, exception_type, exception_value, traceback) -> bool:
        """
        Close the row after leaving the `with` statement.

        :param exception_type: ignored
        :param exception_value: ignored
        :param traceback: ignored
        :returns: `True` to suppress an exception, `False` to rethrow it
        """
        # noinspection PyProtectedMember
        self.__owner._end_row(self.__header_mode)
        return exception_type is None
