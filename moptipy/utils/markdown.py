"""The markdown text format driver."""

from io import TextIOBase
from typing import Final

from moptipy.utils.formatted_string import NUMBER, NAN, \
    POSITIVE_INFINITY, NEGATIVE_INFINITY, TEXT
from moptipy.utils.text_format import TextFormatDriver


class Markdown(TextFormatDriver):
    r"""
    The markdown text driver.

    >>> from io import StringIO
    >>> from moptipy.utils.formatted_string import FormattedStr
    >>> from moptipy.utils.table import Table
    >>> s = StringIO()
    >>> md = Markdown.instance()
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
    >>> print(f"'{s.getvalue()}'")
    '|**1**|`2`|*3*|
    |:--|--:|:-:|
    |a|b|c|
    |d|e|f|
    '
    """

    def end_table_header(self, stream: TextIOBase, cols: str) -> None:
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

    def end_table_row(self, stream: TextIOBase, cols: str,
                      section_index: int, row_index: int) -> None:
        """End a row in a Markdown table."""
        stream.write("|\n")

    def begin_table_cell(self, stream: TextIOBase, cols: str,
                         section_index: int, row_index: int,
                         col_index: int) -> None:
        """Begin a Markdown table cell."""
        stream.write("|")

    def text(self, stream: TextIOBase, text: str, bold: bool, italic: bool,
             code: bool, number_mode: int) -> None:
        """Print a text string."""
        if len(text) <= 0:
            return
        if bold:
            stream.write("**")
        if italic:
            stream.write("*")
        if code:
            stream.write("`")

        if number_mode == TEXT:
            stream.write(text)
        elif number_mode == NUMBER:
            i: int = text.find('e')
            if i < 0:
                i = text.find('E')
            if i > 0:
                stream.write(f"{text[:i]}\\*10^{text[i + 1:]}^")  # \u00D7
            else:
                stream.write(text)
        elif number_mode == NAN:
            stream.write(r"$\emptyset$")  # \u2205
        elif number_mode == POSITIVE_INFINITY:
            stream.write(r"$\infty$")  # \u221E
        elif number_mode == NEGATIVE_INFINITY:
            stream.write(r"$-\infty$")  # -\u221E
        else:
            raise ValueError(f"invalid number mode {number_mode}.")

        if code:
            stream.write("`")
        if italic:
            stream.write("*")
        if bold:
            stream.write("**")

    def __str__(self):
        """
        Get the appropriate file suffix.

        :returns: the file suffix
        :retval 'md': always
        """
        return "md"

    @staticmethod
    def instance() -> 'Markdown':
        """
        Get the markdown format singleton instance.

        :returns: the singleton instance of the Markdown format
        """
        attr: Final[str] = "_instance"
        func: Final = Markdown.instance
        if not hasattr(func, attr):
            setattr(func, attr, Markdown())
        return getattr(func, attr)
