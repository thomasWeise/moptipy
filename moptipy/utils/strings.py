"""Routines for handling strings."""

from re import compile as _compile
from re import sub
from typing import Final, Iterable, Pattern

from pycommons.strings.string_conv import float_to_str, num_to_str, str_to_num
from pycommons.strings.tools import replace_str
from pycommons.types import type_error


def num_to_str_for_name(x: int | float) -> str:
    """
    Convert a float to a string for use in a component name.

    This function can be inverted by applying :func:`name_str_to_num`.

    :param x: the float
    :returns: the string

    >>> num_to_str_for_name(1.3)
    '1d3'
    >>> num_to_str_for_name(1.0)
    '1'
    >>> num_to_str_for_name(-7)
    'm7'
    >>> num_to_str_for_name(-6.32)
    'm6d32'
    >>> num_to_str_for_name(-1e-5)
    'm1em5'
    """
    return num_to_str(x).replace(".", DECIMAL_DOT_REPLACEMENT) \
        .replace("-", MINUS_REPLACEMENT)


def name_str_to_num(s: str) -> int | float:
    """
    Convert a string from a name to a number.

    This function is the inverse of :func:`num_to_str_for_name`.

    :param s: the string from the name
    :returns: an integer or float, depending on the number represented by
        `s`

    >>> name_str_to_num(num_to_str_for_name(1.1))
    1.1
    >>> name_str_to_num(num_to_str_for_name(1))
    1
    >>> name_str_to_num(num_to_str_for_name(-5e3))
    -5000
    >>> name_str_to_num(num_to_str_for_name(-6e-3))
    -0.006
    >>> name_str_to_num(num_to_str_for_name(100.0))
    100
    >>> name_str_to_num(num_to_str_for_name(-1e-4))
    -0.0001
    """
    return str_to_num(s.replace(MINUS_REPLACEMENT, "-")
                      .replace(DECIMAL_DOT_REPLACEMENT, "."))


#: the internal table for converting normal characters to unicode superscripts
__SUPERSCRIPT: Final = str.maketrans({
    # numbers from 0 to 9
    0x30: 0x2070, 0x31: 0x00b9, 0x32: 0x00b2, 0x33: 0x00b3, 0x34: 0x2074,
    0x35: 0x2075, 0x36: 0x2076, 0x37: 0x2077, 0x38: 0x2078, 0x39: 0x2079,
    # +/-/=/(/)
    0x2b: 0x207A, 0x2d: 0x207b, 0x3d: 0x207c, 0x28: 0x207d, 0x29: 0x207e,
    # lower case letters
    0x61: 0x1d43, 0x62: 0x1d47, 0x63: 0x1d9c, 0x64: 0x1d48, 0x65: 0x1d49,
    0x66: 0x1da0, 0x67: 0x1d4d, 0x6b: 0x1d4f, 0x6c: 0x1da9, 0x6d: 0x1d50,
    0x6e: 0x207f, 0x6f: 0x1d52, 0x70: 0x1d56, 0x74: 0x1d57, 0x75: 0x1d58,
    0x76: 0x1d5b, 0x7a: 0x1dbb,
})


def superscript(s: str) -> str:
    """
    Transform a string into Unicode-based superscript.

    :param s: the string
    :returns: the string in superscript

    >>> superscript("a0=4(e)")
    '\u1d43\u2070\u207c\u2074\u207d\u1d49\u207e'
    """
    if not isinstance(s, str):
        raise type_error(s, "s", str)
    return s.translate(__SUPERSCRIPT)


def beautify_float_str(s: str | float) -> str:
    """
    Beautify the string representation of a float.

    This function beautifies the string representation of a float by using
    unicode superscripts for exponents.

    :param s: either a `float` or the string representation of a `float`
    :return: the beautified string representation

    >>> beautify_float_str('0.0')
    '0.0'
    >>> beautify_float_str('1e12')
    '1\u00d710\u00b9\u00b2'
    >>> beautify_float_str('1e-3')
    '1\u00d710\u207b\u00b3'
    >>> beautify_float_str('inf')
    '\u221e'
    >>> beautify_float_str('-inf')
    '-\u221e'
    >>> beautify_float_str('nan')
    '\u2205'
    """
    if isinstance(s, float):
        s = float_to_str(s)
    if not isinstance(s, str):
        raise type_error(s, "s", str)
    s = s.strip().lower()
    if s in ("+inf", "inf"):
        return "\u221e"
    if s == "-inf":
        return "-\u221e"
    if s == "nan":
        return "\u2205"
    eidx: int = s.find("e")
    if eidx < 0:
        return s
    return f"{s[:eidx]}\u00d710{s[eidx + 1:].translate(__SUPERSCRIPT)}"


def __replace_double(replace: str, src: str) -> str:
    """
    Replace any double-occurrence of a string with a single occurrence.

    :param replace: the string to replace
    :param src: the source string
    :returns: the updated string
    """
    return replace_str(replace + replace, replace, src)


#: the separator of different filename parts
PART_SEPARATOR: Final[str] = "_"
#: the replacement for "." in a file name
DECIMAL_DOT_REPLACEMENT: Final[str] = "d"
#: the replacement for "-" in a file name
MINUS_REPLACEMENT: Final[str] = "m"
#: the replacement for "+" in a file name
PLUS_REPLACEMENT: Final[str] = "p"

#: a pattern used during name sanitization
__PATTERN_SPACE_BEFORE_MINUS: Final[Pattern] = _compile(r"[^\w\s-]")
#: the multiple-whitespace pattern
__PATTERN_MULTIPLE_WHITESPACE: Final[Pattern] = _compile(r"\s+")


def sanitize_name(name: str) -> str:
    """
    Sanitize a name in such a way that it can be used as path component.

    >>> sanitize_name(" hello world ")
    'hello_world'
    >>> sanitize_name(" 56.6-455 ")
    '56d6m455'
    >>> sanitize_name(" _ i _ am _ funny   --6 _ ")
    'i_am_funny_m6'

    :param name: the name that should be sanitized
    :return: the sanitized name
    :raises ValueError: if the name is invalid or empty
    :raises TypeError: if the name is `None` or not a string
    """
    if not isinstance(name, str):
        raise type_error(name, "name", str)
    orig_name = name
    name = name.strip()
    name = __replace_double("-", name).replace("+", PLUS_REPLACEMENT)
    name = __replace_double("+", name).replace("-", MINUS_REPLACEMENT)
    name = __replace_double("_", name)
    name = __replace_double(".", name).replace(".", DECIMAL_DOT_REPLACEMENT)

    name = sub(__PATTERN_SPACE_BEFORE_MINUS, "", name)
    name = sub(__PATTERN_MULTIPLE_WHITESPACE, PART_SEPARATOR, name)
    name = __replace_double("_", name)

    if name.startswith("_"):
        name = name[1:]

    if name.endswith("_"):
        name = name[:len(name) - 1]

    if len(name) <= 0:
        raise ValueError(
            f"Sanitized name must not become empty, but {orig_name!r} does.")

    return name


def sanitize_names(names: Iterable[str]) -> str:
    """
    Sanitize a set of names.

    >>> sanitize_names(["", " sdf ", "", "5-3"])
    'sdf_5m3'
    >>> sanitize_names([" a ", " b", " c", "", "6", ""])
    'a_b_c_6'

    :param names: the list of names.
    :return: the sanitized name
    """
    return PART_SEPARATOR.join([
        sanitize_name(name) for name in names if len(name) > 0])
