"""Infrastructure to created structured text, like Markdown and LaTeX."""

from io import TextIOBase
from typing import Final

from moptipy.utils.path import Path
from moptipy.utils.types import type_error
from moptipy.utils.lang import Lang


class FormattedStr(str):
    """A subclass of `str` capable of holding formatting information."""

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
        :param bold: should the format be bold face?
        :param italic: should the format be italic face?
        :param code: should the format be code face?
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

    @staticmethod
    def add_format(s: str, bold: bool = False, italic: bool = False,
                   code: bool = False) -> str:
        """
        Add the given format to the specified string.

        :param s: the string
        :param bold: should the format be bold face?
        :param italic: should the format be italic face?
        :param code: should the format be code face?
        """
        if isinstance(s, FormattedStr):
            bold = bold or s.bold
            italic = italic or s.italic
            code = code or s.code
            if (bold != s.bold) or (italic != s.italic) or (code != s.code):
                return FormattedStr(s, bold, italic, code)
            return s
        if not isinstance(s, str):
            raise type_error(s, "s", str)
        if bold or italic or code:
            return FormattedStr(s, bold, italic, code)
        return s


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

    def end_table_row(self, stream: TextIOBase, cols: str,
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

    def begin_table_cell(self, stream: TextIOBase, cols: str,
                         section_index: int, row_index: int,
                         col_index: int) -> None:
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

    def end_table_cell(self, stream: TextIOBase, cols: str,
                       section_index: int, row_index: int,
                       col_index: int) -> None:
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

    def text(self, stream: TextIOBase, text: str, bold: bool, italic: bool,
             code: bool) -> None:
        """
        Write a chunk of text.

        :param stream: the stream to write to
        :param text: the text to write
        :param bold: is the text in bold face?
        :param italic: is the text in italic face?
        :param code: is the text in code face?
        """

    def render_numeric_exponent(self, e: str) -> str:
        """
        Render the exponent of a number.

        This function is for use in conjunction with
            :func:`moptipy.utils.strings.numbers_to_strings`.

        :param e: the exponent
        :returns: a rendered string
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
