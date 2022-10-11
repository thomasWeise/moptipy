"""Strings that carry format information."""

from math import isnan, nan, inf
from typing import Final
from typing import Optional, Union

from moptipy.utils.types import type_error

#: the formatted string represents normal text
TEXT: Final[int] = 0
#: the formatted string represents a number
NUMBER: Final[int] = 1
#: the formatted string represents NaN, i.e., "not a number"
NAN: Final[int] = 2
#: the formatted string represents positive infinity
POSITIVE_INFINITY: Final[int] = 3
#: the formatted string represents negative infinity
NEGATIVE_INFINITY: Final[int] = 4


class FormattedStr(str):
    """
    A subclass of `str` capable of holding formatting information.

    This is a version of string that also stores format information that can,
    for example, be used by a text format driver when typesetting text.
    Instances of this class can also be used as normal strings and be printed
    to the console without any issue. However, they also hold information
    about whether the text should be :attr:`bold`, :attr:`italic`, or rendered
    in a monospace :attr:`code` font. Furthermore, if a number or numerical
    string is represented as a formatted string, the field :attr:`number_mode`
    will be non-zero. If it is `TEXT=1`, the string is a normal number, if it
    is `NAN=2`, the string is "nan", if it is `POSITIVE_INFINITY=3`, then the
    string is `inf`, and if it is `NEGATIVE_INFINITY=4`, the string is `-inf`.
    These values permit a text driver, for example, to replace the special
    numeric values with unicode constants or certain commands. It may also
    choose to replace floating point number of the form `1.5E23` with
    something like `1.5*10^23`. It can do so because a non-zero
    :attr:`number_mode` indicates that the string is definitely representing a
    number and that numbers in the string did not just occur for whatever
    other reason.
    """

    #: should this string be formatted in bold face?
    bold: bool
    #: should this string be formatted in italic face?
    italic: bool
    #: should this string be formatted in code face?
    code: bool
    #: the numeric mode: `TEXT`, `NUMBER`, `NAN`, `POSITIVE_INFINITY`,
    #: or `NEGATIVE_INFINITY`
    number_mode: int

    def __new__(cls, value, bold: bool = False, italic: bool = False,
                code: bool = False, number_mode: int = TEXT):
        """
        Construct the object.

        :param value: the string value
        :param bold: should the format be bold face?
        :param italic: should the format be italic face?
        :param code: should the format be code face?
        :param number_mode: the number mode
        """
        if not isinstance(bold, bool):
            raise type_error(bold, "bold", bool)
        if not isinstance(italic, bool):
            raise type_error(italic, "italic", bool)
        if not isinstance(code, bool):
            raise type_error(code, "code", bool)
        if not isinstance(number_mode, int):
            raise type_error(number_mode, "number_mode", int)
        if (number_mode < TEXT) or (number_mode > NEGATIVE_INFINITY):
            raise ValueError(f"invalid number mode: {number_mode}")
        if bold or italic or code or (number_mode != TEXT):
            ret = super(FormattedStr, cls).__new__(cls, value)
            ret.bold = bold
            ret.italic = italic
            ret.code = code
            ret.number_mode = number_mode
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

        >>> from typing import cast
        >>> st = "abc"
        >>> type(st)
        <class 'str'>
        >>> fs = cast(FormattedStr, FormattedStr.add_format(st, bold=True))
        >>> type(fs)
        <class 'moptipy.utils.formatted_string.FormattedStr'>
        >>> fs.bold
        True
        >>> fs.italic
        False
        >>> fs = cast(FormattedStr, FormattedStr.add_format(fs, italic=True))
        >>> fs.bold
        True
        >>> fs.italic
        True
        >>> fs.number_mode
        0
        """
        if isinstance(s, FormattedStr):
            bold = bold or s.bold
            italic = italic or s.italic
            code = code or s.code
            if (bold != s.bold) or (italic != s.italic) or (code != s.code):
                return FormattedStr(s, bold, italic, code, s.number_mode)
            return s
        if not isinstance(s, str):
            raise type_error(s, "s", str)
        if bold or italic or code:
            return FormattedStr(s, bold, italic, code, TEXT)
        return s

    @staticmethod
    def number(number: Optional[Union[int, float, str]]) -> 'FormattedStr':
        """
        Create a formatted string representing a number.

        :param number: the original number or numeric string
        :return: the formatted string representing it

        >>> FormattedStr.number(inf)
        'inf'
        >>> FormattedStr.number(inf) is _PINF
        True
        >>> FormattedStr.number(inf).number_mode
        3
        >>> FormattedStr.number(-inf)
        '-inf'
        >>> FormattedStr.number(-inf) is _NINF
        True
        >>> FormattedStr.number(-inf).number_mode
        4
        >>> FormattedStr.number(nan)
        'nan'
        >>> FormattedStr.number(nan) is _NAN
        True
        >>> FormattedStr.number(nan).number_mode
        2
        >>> FormattedStr.number(123)
        '123'
        >>> FormattedStr.number(123e3)
        '123000.0'
        >>> FormattedStr.number(123e3).number_mode
        1
        """
        if not isinstance(number, (int, str)):
            if isinstance(number, float):
                if isnan(number):
                    return _NAN
                if number >= inf:
                    return _PINF
                if number <= -inf:
                    return _NINF
            else:
                raise type_error(number, "number", (float, int, str))
        return FormattedStr(str(number), False, False, False, NUMBER)


#: the constant for not-a-number
_NAN: Final[FormattedStr] = FormattedStr(str(nan), False, False, False, NAN)
#: the constant for positive infinity
_PINF: Final[FormattedStr] = FormattedStr(str(inf), False, False, False,
                                          POSITIVE_INFINITY)
#: the constant for negative infinity
_NINF: Final[FormattedStr] = FormattedStr(str(-inf), False, False, False,
                                          NEGATIVE_INFINITY)
