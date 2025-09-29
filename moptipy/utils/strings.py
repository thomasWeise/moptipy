"""Routines for handling strings."""

from re import compile as _compile
from re import sub
from typing import Final, Iterable, Pattern

from pycommons.strings.chars import superscript
from pycommons.strings.string_conv import float_to_str, num_to_str, str_to_num
from pycommons.strings.string_tools import replace_str
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
    if s in {"+inf", "inf"}:
        return "\u221e"
    if s == "-inf":
        return "-\u221e"
    if s == "nan":
        return "\u2205"
    eidx: int = s.find("e")
    if eidx < 0:
        return s
    return f"{s[:eidx]}\u00d710{superscript(s[eidx + 1:])}"


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

    name = name.removeprefix("_").removesuffix("_")
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
