"""The HTML text format driver."""

from io import TextIOBase
from typing import Final

from moptipy.utils.text_format import TextFormatDriver

#: the default border style
_BORDER: Final[str] = "1pt solid black"


class HTML(TextFormatDriver):
    """The HTML text driver."""

    def begin_table_body(self, stream: TextIOBase, cols: str) -> None:
        """Write the beginning of the table body."""
        stream.write(
            f'<table style="border:{_BORDER};border-collapse:collapse">')

    def end_table_body(self, stream: TextIOBase, cols: str) -> None:
        """Write the ending of the table body."""
        stream.write("</tbody></table>")

    def begin_table_header(self, stream: TextIOBase, cols: str) -> None:
        """Begin the header of an HTML table."""
        stream.write("<thead>")

    def end_table_header(self, stream: TextIOBase, cols: str) -> None:
        """End the header of an HTML table."""
        stream.write("</thead><tbody>")

    def begin_table_row(self, stream: TextIOBase, cols: str,
                        section_index: int, row_index: int) -> None:
        """Begin a row in an HTML table."""
        stream.write("<tr>")

    def end_table_row(self, stream: TextIOBase, cols: str,
                      section_index: int, row_index: int) -> None:
        """End a row in a HTML table."""
        stream.write("</tr>")

    def begin_table_cell(self, stream: TextIOBase, cols: str,
                         section_index: int, row_index: int,
                         col_index: int) -> None:
        """Begin an HTML table cell."""
        style: str = cols[col_index]
        style = "center" if style == "c" \
            else "left" if style == "l" else "right"
        style = f"padding:3pt;text-align:{style}"
        if col_index < (len(cols) - 1):
            style = f"{style};border-right:{_BORDER}"
        if row_index <= 0 <= section_index:
            style = f"{style};border-top:{_BORDER}"
        cell: str = "th" if (section_index < 0) else "td"
        stream.write(f'<{cell} style="{style}">')

    def end_table_cell(self, stream: TextIOBase, cols: str,
                       section_index: int, row_index: int,
                       col_index: int) -> None:
        """End an HTML table cell."""
        stream.write("</th>" if (section_index < 0) else "</td>")

    def text(self, stream: TextIOBase, text: str, bold: bool, italic: bool,
             code: bool) -> None:
        """Print a text string."""
        if len(text) <= 0:
            return
        styles: str = ""
        if bold:
            styles = "font-weight:bold;"
        if italic:
            styles += "font-style:italic;"
        if code:
            styles += "font-family:monospace;"
        if len(styles) > 0:
            stream.write(f'<span style="{styles[0:-1]}">')
        stream.write(text)
        if len(styles) > 0:
            stream.write("</span>")

    def __str__(self):
        """
        Get the appropriate file suffix.

        :returns: the file suffix
        :retval 'html': always
        """
        return "html"

    def render_numeric_exponent(self, e: str) -> str:
        r"""
        Render the numerical exponent in markdown.

        :param e: the exponent
        :returns: the rendered exponent
        :retval: `*10\textsuperscript{e}`
        """
        return f"*10<sup>{e}</sup>"

    @staticmethod
    def instance() -> 'HTML':
        """
        Get the HTML format singleton instance.

        :returns: the singleton instance of the HTML format
        """
        attr: Final[str] = "_instance"
        func: Final = HTML.instance
        if not hasattr(func, attr):
            setattr(func, attr, HTML())
        return getattr(func, attr)
