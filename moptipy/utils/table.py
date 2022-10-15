"""Classes for printing tables in a text format."""

from contextlib import AbstractContextManager
from io import TextIOBase
from typing import Final, Optional, Iterable, Union, Callable, List

from moptipy.utils.formatted_string import FormattedStr, TEXT
from moptipy.utils.text_format import TextFormatDriver, MODE_NORMAL, \
    MODE_TABLE_HEADER, MODE_SECTION_HEADER
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
    :meth:`Rows.row` and :class:`Row`).
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
        self.columns: Final[str] = cols
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

    def _begin_rows(self, mode: int):
        """
        Start a set of rows.

        :param mode: the mode of the rows, will be one of `MODE_NORMAL`,
            `MODE_TABLE_HEADER`, or `MODE_SECTION_HEADER`
        """
        if self.__stream is None:
            raise ValueError("table already closed, cannot start section.")

        if mode == MODE_NORMAL:
            if self.__header_state <= 1:
                raise ValueError("cannot start section before table body.")
            if self.__section_state == 1:
                raise ValueError("cannot start section inside section.")
            if self.__section_header_state == 1:
                raise ValueError(
                    "cannot start section inside or after section header.")
            if self.__row_state == 1:
                raise ValueError("cannot start section inside row.")
            self.__section_state = 1
            self.__section_header_state = 0
            self.__driver.begin_table_section(self.__stream, self.columns,
                                              self.__section_index)
            self.__section_index += 1

        elif mode == MODE_TABLE_HEADER:
            if self.__header_state >= 1:
                raise ValueError(
                    "cannot start table header inside or after table header.")
            if self.__section_state >= 1:
                raise ValueError(
                    "cannot start table header inside or after section.")
            if self.__section_header_state >= 1:
                raise ValueError("cannot start table header inside or "
                                 "after section header.")
            if self.__row_state >= 1:
                raise ValueError("cannot start table header inside row.")
            self.__header_state = 1
            self.__driver.begin_table_header(self.__stream, self.columns)

        elif mode == MODE_SECTION_HEADER:
            if self.__header_state <= 1:
                raise ValueError(
                    "cannot start section header before or in table header.")
            if self.__section_state != 1:
                raise ValueError(
                    "cannot start section header outside section.")
            if self.__section_header_state > 1:
                raise ValueError(
                    "cannot start section header after section header.")
            if self.__row_state == 1:
                raise ValueError(
                    "cannot start section header inside row.")
            self.__section_header_state = 1
            self.__driver.begin_table_section_header(
                self.__stream, self.columns, self.__section_index)
        else:
            raise ValueError(f"invalid row group mode: {mode}")

        self.__row_index = 0
        self.__row_state = 0

    def _end_rows(self, mode: int):
        """
        End a set of rows.

        :param mode: the mode of the rows, will be one of `MODE_NORMAL`,
            `MODE_TABLE_HEADER`, or `MODE_SECTION_HEADER`
        """
        if self.__stream is None:
            raise ValueError("table already closed, cannot end section.")

        if mode == MODE_NORMAL:
            if self.__header_state <= 1:
                raise ValueError(
                    "cannot end section before end of table header.")
            if self.__section_state != 1:
                raise ValueError("cannot end section outside section.")
            if self.__section_header_state == 1:
                raise ValueError("cannot end section inside section header.")
            if self.__row_state == 1:
                raise ValueError("cannot end section inside of row.")
            if (self.__row_index <= 0) or (self.__row_state < 2):
                raise ValueError("cannot end section before writing any row.")
            self.__section_state = 2
            self.__driver.end_table_section(
                self.__stream, self.columns, self.__section_index,
                self.__row_index)

        elif mode == MODE_TABLE_HEADER:
            if self.__header_state != 1:
                raise ValueError(
                    "cannot end table header outside table header.")
            if self.__section_state != 0:
                raise ValueError(
                    "cannot end table header inside or after section.")
            if self.__section_header_state >= 1:
                raise ValueError(
                    "cannot end table header inside or after section header.")
            if self.__row_state == 1:
                raise ValueError("cannot end table header inside row.")
            if (self.__row_state < 2) or (self.__row_index <= 0):
                raise ValueError("cannot end table header before header row.")
            self.__header_state = 2
            self.__driver.end_table_header(self.__stream, self.columns)

        elif mode == MODE_SECTION_HEADER:
            if self.__header_state < 2:
                raise ValueError(
                    "cannot end section header before table body.")
            if self.__section_state != 1:
                raise ValueError(
                    "cannot start section header outside section.")
            if self.__section_header_state != 1:
                raise ValueError(
                    "cannot end section header only inside section header.")
            if self.__row_state == 1:
                raise ValueError("cannot end section header inside row.")
            if (self.__row_state < 2) or (self.__row_index <= 0):
                raise ValueError(
                    "cannot end section header before section header row.")
            self.__section_header_state = 2
            self.__driver.end_table_section_header(
                self.__stream, self.columns, self.__section_index)
        else:
            raise ValueError(f"invalid row group mode: {mode}")

        self.__row_index = 0

    def _begin_row(self, mode: int):
        """
        Start a row.

        :param mode: the mode of the row, will be one of `MODE_NORMAL`,
            `MODE_TABLE_HEADER`, or `MODE_SECTION_HEADER`
        """
        if self.__stream is None:
            raise ValueError("table already closed, cannot start row.")
        if self.__row_state == 1:
            raise ValueError("cannot start row inside row.")

        if mode == MODE_NORMAL:
            if self.__section_state != 1:
                raise ValueError("can only start section row in section.")
            if self.__section_header_state == 1:
                self.__section_header_state = 2
                self.__row_index = 0
                self.__driver.end_table_section_header(
                    self.__stream, self.columns, self.__section_index)
        elif mode == MODE_TABLE_HEADER:
            if self.__header_state != 1:
                raise ValueError("can only start header row in table header.")
        elif mode == MODE_SECTION_HEADER:
            if self.__section_state != 1:
                raise ValueError(
                    "can only start section header row in section.")
            if self.__section_header_state > 1:
                raise ValueError(
                    "cannot start section header row after section header.")
            if self.__section_header_state < 1:
                if self.__row_index > 0:
                    raise ValueError(
                        "cannot start section header after section row.")
                self.__section_header_state = 1
                self.__driver.begin_table_section_header(
                    self.__stream, self.columns, self.__section_index)

        else:
            raise ValueError(f"invalid row mode: {mode}")

        self.__driver.begin_table_row(
            self.__stream, self.columns, self.__section_index,
            self.__row_index, mode)
        self.__row_index += 1
        self.__row_state = 1
        self.__col_index = 0

    def _end_row(self, mode: int):
        """
        End a row.

        :param mode: the mode of the row, will be one of `MODE_NORMAL`,
            `MODE_TABLE_HEADER`, or `MODE_SECTION_HEADER`
        """
        if self.__stream is None:
            raise ValueError("table already closed, cannot start row.")

        if not (MODE_NORMAL <= mode <= MODE_SECTION_HEADER):
            raise ValueError(f"invalid row mode {mode}.")
        if self.__header_state == 0:
            raise ValueError(
                "cannot end row before table header.")
        if self.__section_state >= 2:
            raise ValueError("cannot end row after section.")
        if self.__row_state != 1:
            raise ValueError("can end row only inside row.")
        if self.__col_index != len(self.columns):
            raise ValueError(
                f"cannot end row after {self.__col_index} columns for table "
                f"with column definition {self.columns}.")
        self.__driver.end_table_row(self.__stream, self.columns,
                                    self.__section_index, self.__row_index)
        self.__row_state = 2

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
        if self.__row_state != 1:
            raise ValueError(
                "cells only permitted inside rows.")
        col_index: Final[int] = self.__col_index
        if col_index >= len(self.columns):
            raise ValueError(
                f"cannot add cell after {col_index} columns for table "
                f"with column definition {self.columns}.")

        mode: Final[int] = MODE_TABLE_HEADER if self.__header_state == 1 \
            else (MODE_SECTION_HEADER if self.__section_header_state == 1
                  else MODE_NORMAL)

        self.__driver.begin_table_cell(
            self.__stream, self.columns, self.__section_index,
            self.__row_index, col_index, mode)
        self.__col_index = col_index + 1

        def __printit(st, strm: TextIOBase = self.__stream,
                      wrt: Callable[[TextIOBase, str, bool, bool, bool, int],
                                    None] = self.__driver.text) \
                -> None:
            if st is None:
                return
            if isinstance(st, str):
                if isinstance(st, FormattedStr):
                    wrt(strm, st, st.bold, st.italic, st.code, st.mode)
                else:
                    wrt(strm, st, False, False, False, TEXT)
            elif isinstance(st, Iterable):
                for ss in st:
                    __printit(ss)
            else:
                raise type_error(st, "text", (Iterable, str))

        __printit(text)

        self.__driver.end_table_cell(
            self.__stream, self.columns, self.__section_index,
            self.__row_index, col_index, mode)

    def header(self) -> 'Rows':
        """
        Construct the header of the table.

        :returns: a new managed header row
        """
        return Rows(self, MODE_TABLE_HEADER)

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
        self.__driver.begin_table_body(self.__stream, self.columns)
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
            self.__driver.end_table_body(self.__stream, self.columns)
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


class Rows(AbstractContextManager):
    """A set of table rows."""

    def __init__(self, owner: Table, mode: int):
        """
        Initialize the row section.

        :param owner: the owning table
        :param mode: the mode of the row group
        """
        if not isinstance(owner, Table):
            raise type_error(owner, "owner", Table)
        #: the owner
        self._owner: Final[Table] = owner
        if not isinstance(mode, int):
            raise type_error(mode, "mode", int)
        if not (MODE_NORMAL <= mode <= MODE_SECTION_HEADER):
            raise ValueError(f"wrong mode: {mode}, must be in "
                             f"{MODE_NORMAL}..{MODE_SECTION_HEADER}")
        #: the rows mode
        self._mode: Final[int] = mode

    def __enter__(self):
        """
        Enter the row section in a `with` statement.

        :return: `self`
        """
        # noinspection PyProtectedMember
        self._owner._begin_rows(self._mode)
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
        self._owner._end_rows(self._mode)
        return exception_type is None

    def row(self) -> 'Row':
        """
        Create a row.

        :return: the new row
        """
        return Row(self._owner, self._mode)

    def full_row(self, cells: Iterable[Optional[str]]) -> None:
        """
        Print a complete row with a single call.

        :param cells: the iterable of strings for the cells.
        """
        if not isinstance(cells, Iterable):
            raise type_error(cells, "cells", Iterable)
        with self.row() as row:
            for i, cell in enumerate(cells):
                if cell is not None:
                    if not isinstance(cell, str):
                        raise type_error(cell, f"cell[{i}]", str)
                row.cell(cell)

    def cols(self, cols: List[List[Optional[str]]]):
        """
        Print cells and rows column-by-column.

        :param cols: an array which contains one list per column of the table.
        """
        if not isinstance(cols, list):
            raise type_error(cols, "cols", list)

        columns: Final[str] = self._owner.columns
        if len(cols) != len(columns):
            raise ValueError(
                f"expected {len(columns)} columns ({columns}), "
                f"but cols has length {len(cols)}.")
        max_rows = max(len(col) for col in cols)
        if max_rows <= 0:
            raise ValueError("There are no rows in the cols array?")
        for rowi in range(max_rows):
            with self.row() as row:
                for col in cols:
                    row.cell(None if rowi >= len(col) else col[rowi])


class Section(Rows):
    """A table section is a group of rows, potentially with a header."""

    def __init__(self, owner: Table):
        """
        Initialize the row section.

        :param owner: the owning table
        """
        super().__init__(owner, MODE_NORMAL)

    def header(self) -> 'Rows':
        """
        Print the section header.

        :return: the header row
        """
        return Rows(self._owner, MODE_SECTION_HEADER)


class Row(AbstractContextManager):
    """A row class."""

    def __init__(self, owner: Table, mode: int):
        """
        Initialize the row.

        :param owner: the owning table
        :param mode: the header mode
        """
        if not isinstance(owner, Table):
            raise type_error(owner, "owner", Table)
        if not isinstance(mode, int):
            raise type_error(mode, "mode", int)
        if not (MODE_NORMAL <= mode <= MODE_SECTION_HEADER):
            raise ValueError(f"wrong mode: {mode}, must be in "
                             f"{MODE_NORMAL}..{MODE_SECTION_HEADER}")
        #: the rows mode
        self._mode: Final[int] = mode
        #: the owner
        self.__owner: Final[Table] = owner

    def cell(self, text: Optional[Union[str, Iterable[str]]] = None) -> None:
        """
        Render the text of a cell.

        As parameter `text`, you can provide either a string or a sequence of
        strings. You can also provide an instance of
        :class:`moptipy.utils.formatted_string.FormattedStr` or a sequence
        thereof. This allows you to render formatted text in a natural
        fashion.

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
        self.__owner._begin_row(self._mode)
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
        self.__owner._end_row(self._mode)
        return exception_type is None
