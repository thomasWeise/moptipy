"""The latex text format driver."""

from io import TextIOBase
from typing import Final

from moptipy.utils.text_format import TextFormatDriver


class LaTeX(TextFormatDriver):
    r"""
    The LaTeX text driver.

    >>> from io import StringIO
    >>> from moptipy.utils.table import Table
    >>> from moptipy.utils.text_format import FormattedStr
    >>> s = StringIO()
    >>> latex = LaTeX.instance()
    >>> print(str(latex))
    tex
    >>> with Table(s, "lrc", latex) as t:
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
    '\begin{tabular}{lrc}%
    \hline%
    {\textbf{1}}&{\texttt{2}}&{\textit{3}}\\%
    \hline%
    a&b&c\\%
    d&e&f\\%
    \hline%
    \end{tabular}%
    '
    """

    def begin_table_body(self, stream: TextIOBase, cols: str) -> None:
        """Write the beginning of the table body."""
        stream.write(f"\\begin{{tabular}}{{{cols}}}%\n")

    def end_table_body(self, stream: TextIOBase, cols: str) -> None:
        """Write the ending of the table body."""
        stream.write("\\end{tabular}%\n")

    def begin_table_header(self, stream: TextIOBase, cols: str) -> None:
        """Begin the header of a LaTeX table."""
        stream.write("\\hline%\n")

    def end_table_header(self, stream: TextIOBase, cols: str) -> None:
        """End the header of a LaTeX table."""
        stream.write("\\hline%\n")

    def end_table_section(self, stream: TextIOBase, cols: str,
                          section_index: int, n_rows: int) -> None:
        """End a table section."""
        stream.write("\\hline%\n")

    def end_table_section_header(self, stream: TextIOBase, cols: str,
                                 section_index: int) -> None:
        """End a table section header."""
        stream.write("\\hline%\n")

    def end_table_row(self, stream: TextIOBase, cols: str,
                      section_index: int, row_index: int) -> None:
        """End a row in a LaTeX table."""
        stream.write("\\\\%\n")

    def begin_table_cell(self, stream: TextIOBase, cols: str,
                         section_index: int, row_index: int,
                         col_index: int) -> None:
        """Begin a LaTeX table cell."""
        if col_index > 0:
            stream.write("&")

    def text(self, stream: TextIOBase, text: str, bold: bool, italic: bool,
             code: bool) -> None:
        """Print a text string."""
        if len(text) <= 0:
            return
        if bold:
            stream.write("{\\textbf{")
        if italic:
            stream.write("{\\textit{")
        if code:
            stream.write("{\\texttt{")
        stream.write(text.replace("_", "\\_"))
        if code:
            stream.write("}}")
        if italic:
            stream.write("}}")
        if bold:
            stream.write("}}")

    def __str__(self):
        """
        Get the appropriate file suffix.

        :returns: the file suffix
        :retval 'tex': always
        """
        return "tex"

    def render_numeric_exponent(self, e: str) -> str:
        r"""
        Render the numerical exponent in LaTeX.

        :param e: the exponent
        :returns: the rendered exponent
        :retval: `\hspace*{0.15em}*\hspace*{0.1em}10\textsuperscript{e}`
        """
        return f"\\hspace*{{0.15em}}*\\hspace*{{0.1em}}" \
               f"10\\textsuperscript{{{e}}}"

    @staticmethod
    def instance() -> 'LaTeX':
        """
        Get the LaTeX format singleton instance.

        :returns: the singleton instance of the LaTeX format
        """
        attr: Final[str] = "_instance"
        func: Final = LaTeX.instance
        if not hasattr(func, attr):
            setattr(func, attr, LaTeX())
        return getattr(func, attr)
