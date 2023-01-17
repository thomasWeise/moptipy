"""The latex text format driver."""

from io import TextIOBase
from typing import Final

from moptipy.utils.formatted_string import (
    NAN,
    NEGATIVE_INFINITY,
    NUMBER,
    POSITIVE_INFINITY,
    SPECIAL,
    TEXT,
)
from moptipy.utils.text_format import TextFormatDriver

#: the exponent prefix
_EPREFIX = r"\hspace*{0.15em}*\hspace*{0.1em}10\textsuperscript{"

#: special characters in LaTeX
SPECIAL_CHARS: Final[dict[str, str]] = {
    "\u2205": r"$\emptyset$",
    "\u221E": r"$\infty$",
    "-\u221E": r"$-\infty$",
    "inf": r"$\infty$",
    "-inf": r"$-\infty$",
    "nan": r"$\emptyset$",
    "\u03b1": r"$\alpha$",
    "\u2014": "---",
}


class LaTeX(TextFormatDriver):
    r"""
    The LaTeX text driver.

    >>> from io import StringIO
    >>> from moptipy.utils.formatted_string import FormattedStr
    >>> from moptipy.utils.table import Table
    >>> s = StringIO()
    >>> latex = LaTeX.instance()
    >>> print(str(latex))
    tex
    >>> with Table(s, "lrc", latex) as t:
    ...     with t.header() as header:
    ...         with header.row() as h:
    ...             h.cell(FormattedStr("1", bold=True))
    ...             h.cell(FormattedStr("2", code=True))
    ...             h.cell(FormattedStr("3", italic=True))
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
                         col_index: int, cell_mode: int) -> None:
        """Begin a LaTeX table cell."""
        if col_index > 0:
            stream.write("&")

    def text(self, stream: TextIOBase, text: str, bold: bool, italic: bool,
             code: bool, mode: int) -> None:
        """Print a text string."""
        if len(text) <= 0:
            return
        if bold:
            stream.write("{\\textbf{")
        if italic:
            stream.write("{\\textit{")
        if code:
            stream.write("{\\texttt{")

        if mode == TEXT:
            stream.write(text.replace("_", "\\_"))
        elif mode == NUMBER:
            i: int = text.find("e")
            if i < 0:
                i = text.find("E")
            if i > 0:
                stream.write(f"{text[:i]}{_EPREFIX}{text[i + 1:]}}}")
            else:
                stream.write(text.replace("_", "\\_"))
        elif mode == NAN:
            stream.write(r"$\emptyset$")
        elif mode == POSITIVE_INFINITY:
            stream.write(r"$\infty$")
        elif mode == NEGATIVE_INFINITY:
            stream.write(r"$-\infty")
        elif mode == SPECIAL:
            s: Final[str] = str(text)
            if s not in SPECIAL_CHARS:
                raise ValueError(f"invalid special character: {s!r}")
            stream.write(SPECIAL_CHARS[s])
        else:
            raise ValueError(f"invalid mode {mode} for text {text!r}.")

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

    @staticmethod
    def instance() -> "LaTeX":
        """
        Get the LaTeX format singleton instance.

        :returns: the singleton instance of the LaTeX format
        """
        attr: Final[str] = "_instance"
        func: Final = LaTeX.instance
        if not hasattr(func, attr):
            setattr(func, attr, LaTeX())
        return getattr(func, attr)
