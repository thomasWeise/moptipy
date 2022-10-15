"""Infrastructure to created structured text, like Markdown and LaTeX."""

from io import TextIOBase
from typing import Final

from moptipy.utils.lang import Lang
from moptipy.utils.path import Path
from moptipy.utils.types import type_error

#: indicates a normal row or cell
MODE_NORMAL: Final[int] = 0
#: indicates a row or cell in the table header
MODE_TABLE_HEADER: Final[int] = 1
#: indicates a row or cell in the section header
MODE_SECTION_HEADER: Final[int] = 2


class TextFormatDriver:
    """
    A base class for text format drivers.

    Table drivers allow us to render the structured text. It used, for
    example, by instances of :class:`~moptipy.utils.table.Table` to write
    tabular data to a text format stream.
    """

    def begin_table_body(self, stream: TextIOBase, cols: str) -> None:
        """
        Write the beginning of the table body.

        :param stream: the stream to write to
        :param cols: the column definition
        """

    def end_table_body(self, stream: TextIOBase, cols: str) -> None:
        """
        Write the ending of the table body.

        :param stream: the stream to write to
        :param cols: the column definition
        """

    def begin_table_header(self, stream: TextIOBase, cols: str) -> None:
        """
        Begin the header row of the table.

        :param stream: the stream to write to
        :param cols: the column definition
        """

    def end_table_header(self, stream: TextIOBase, cols: str) -> None:
        """
        End the header row of the table.

        :param stream: the stream to write to
        :param cols: the column definition
        """

    def begin_table_section(self, stream: TextIOBase, cols: str,
                            section_index: int) -> None:
        """
        Begin a new section of the table.

        :param stream: the stream to write to
        :param cols: the column definition
        :param section_index: the index of the section, `0` for the first
            section
        """

    def end_table_section(self, stream: TextIOBase, cols: str,
                          section_index: int,
                          n_rows: int) -> None:
        """
        End a section of the table.

        :param stream: the stream to write to
        :param cols: the column definition
        :param section_index: the index of the section, `0` for the first
            section
        :param n_rows: the number of rows that were written in the section
        """

    def begin_table_section_header(self, stream: TextIOBase, cols: str,
                                   section_index: int) -> None:
        """
        Begin the header row of a section.

        :param stream: the stream to write to
        :param cols: the column definition
        :param section_index: the index of the section, `0` for the first
            section
        """

    def end_table_section_header(self, stream: TextIOBase, cols: str,
                                 section_index: int) -> None:
        """
        End the header row of a section.

        :param stream: the stream to write to
        :param cols: the column definition
        :param section_index: the index of the section, `0` for the first
            section
        """

    def begin_table_row(self, stream: TextIOBase, cols: str,
                        section_index: int, row_index: int,
                        row_mode: int) -> None:
        """
        Begin a table header row, section row, or normal row in a section.

        :param stream: the stream to write to
        :param cols: the column definition
        :param section_index: the index of the current section, `-1` if we
            are in the table header
        :param row_index: the row index in the section or header
        :param row_mode: the mode of the row, will be one of `MODE_NORMAL`,
            `MODE_TABLE_HEADER`, or `MODE_SECTION_HEADER`
        """

    def end_table_row(self, stream: TextIOBase, cols: str,
                      section_index: int, row_index: int) -> None:
        """
        End a table header row, section row, or normal row in a section.

        :param stream: the stream to write to
        :param cols: the column definition
        :param section_index: the index of the current section
        :param row_index: the row index in the section or header
        """

    def begin_table_cell(self, stream: TextIOBase, cols: str,
                         section_index: int, row_index: int,
                         col_index: int, cell_mode: int) -> None:
        """
        Begin a header cell, section header cell, or normal cell.

        :param stream: the stream to write to
        :param cols: the column definitions
        :param section_index: the index of the current section, `-1` if this
            is a table header cell
        :param row_index: the row index in the section or header
        :param col_index: the column index, `0` for the first column
        :param cell_mode: the mode of the cell, will be one of `MODE_NORMAL`,
            `MODE_TABLE_HEADER`, or `MODE_SECTION_HEADER`
        """

    def end_table_cell(self, stream: TextIOBase, cols: str,
                       section_index: int, row_index: int,
                       col_index: int, cell_mode: int) -> None:
        """
        End a header cell, section header cell, or normal cell.

        :param stream: the stream to write to
        :param cols: the column definitions
        :param section_index: the index of the current section, `-1` if this
            is a table header cell
        :param row_index: the row index in the section or header
        :param col_index: the column index, `0` for the first column
        :param cell_mode: the mode of the cell, will be one of `MODE_NORMAL`,
            `MODE_TABLE_HEADER`, or `MODE_SECTION_HEADER`
        """

    def text(self, stream: TextIOBase, text: str, bold: bool, italic: bool,
             code: bool, mode: int) -> None:
        """
        Write a chunk of text.

        :param stream: the stream to write to
        :param text: the text to write
        :param bold: is the text in bold face?
        :param italic: is the text in italic face?
        :param code: is the text in code face?
        :param mode: the mode of a formatted text piece, see the attribute
            :attr:`~moptipy.utils.formatted_string.FormattedStr.mode` of
            :class:`~moptipy.utils.formatted_string.FormattedStr`
        """

    def filename(self,
                 file_name: str = "file",
                 dir_name: str = ".",
                 use_lang: bool = True) -> Path:
        """
        Get the right filename for this text driver.

        :param file_name: the base file name
        :param dir_name: the base directory
        :param use_lang: should we use the language to define the filename?
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
        if not isinstance(use_lang, bool):
            raise type_error(use_lang, "use_lang", bool)
        out_dir = Path.directory(dir_name)
        suffix = str(self)
        if not isinstance(suffix, str):
            raise type_error(suffix, "result of str(table driver)", str)
        if len(suffix) <= 0:
            raise ValueError(f"invalid driver suffix: '{suffix}'")
        if use_lang:
            file_name = Lang.current().filename(file_name)
        file: Final[Path] = out_dir.resolve_inside(f"{file_name}.{suffix}")
        file.ensure_file_exists()
        return file
