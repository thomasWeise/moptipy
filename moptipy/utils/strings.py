"""Routines for handling strings."""

from math import isnan
from re import MULTILINE, sub
from re import compile as _compile
from typing import Final, Iterable, Pattern

from moptipy.utils.math import __try_int
from moptipy.utils.types import type_error


def float_to_str(x: float) -> str:
    """
    Convert `float` to a string.

    :param x: the floating point value
    :return: the string representation

    >>> float_to_str(1.3)
    '1.3'
    >>> float_to_str(1.0)
    '1'
    >>> float_to_str(1e-5)
    '1e-5'
    """
    if x == 0.0:
        return "0"
    s = repr(x).replace("e-0", "e-")
    if isnan(x):
        raise ValueError(f"{str(s)!r} not permitted.")
    if s.endswith(".0"):
        return s[:-2]
    return s


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
    return str_to_intfloat(s.replace(MINUS_REPLACEMENT, "-")
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


def bool_to_str(value: bool) -> str:
    """
    Convert a Boolean value to a string.

    :param value: the Boolean value
    :return: the string

    >>> print(bool_to_str(True))
    T
    >>> print(bool_to_str(False))
    F
    """
    return "T" if value else "F"


def str_to_bool(value: str) -> bool:
    """
    Convert a string to a boolean value.

    :param value: the string value
    :return: the boolean value

    >>> str_to_bool("T")
    True
    >>> str_to_bool("F")
    False
    >>> try:
    ...     str_to_bool("x")
    ... except ValueError as v:
    ...     print(v)
    Expected 'T' or 'F', but got 'x'.
    """
    if value == "T":
        return True
    if value == "F":
        return False
    raise ValueError(f"Expected 'T' or 'F', but got {str(value)!r}.")


def num_to_str(value: int | float) -> str:
    """
    Transform a numerical type to a string.

    :param value: the value
    :return: the string

    >>> num_to_str(1)
    '1'
    >>> num_to_str(1.5)
    '1.5'
    """
    return str(value) if isinstance(value, int) else float_to_str(value)


def intfloatnone_to_str(val: int | float | None) -> str:
    """
    Convert an integer ot float or `None` to a string.

    :param val: the value
    :return: the string representation
    :rtype: str

    >>> print(repr(intfloatnone_to_str(None)))
    ''
    >>> print(intfloatnone_to_str(12))
    12
    >>> print(intfloatnone_to_str(12.3))
    12.3
    """
    return "" if val is None else num_to_str(val)


def intnone_to_str(val: int | None) -> str:
    """
    Convert an integer or `None` to a string.

    :param val: the value
    :return: the string representation

    >>> print(repr(intnone_to_str(None)))
    ''
    >>> print(intnone_to_str(12))
    12
    """
    return "" if val is None else str(val)


def str_to_intfloat(val: str) -> int | float:
    """
    Convert a string to an int or float.

    :param val: the string value
    :return: the int or float

    >>> print(type(str_to_intfloat("15.0")))
    <class 'int'>
    >>> print(type(str_to_intfloat("15.1")))
    <class 'float'>
    """
    return __try_int(float(val)) if ("e" in val) or \
                                    ("E" in val) or \
                                    ("." in val) or \
                                    ("inf" in val) else int(val)


def str_to_intfloatnone(val: str) -> int | float | None:
    """
    Convert a string to an int or float or None.

    :param val: the string value
    :return: the int or float or None

    >>> print(str_to_intfloatnone(""))
    None
    >>> print(type(str_to_intfloatnone("5.0")))
    <class 'int'>
    >>> print(type(str_to_intfloatnone("5.1")))
    <class 'float'>
    """
    return None if len(val) <= 0 else str_to_intfloat(val)


def str_to_intnone(val: str) -> int | None:
    """
    Convert a string to an int or None.

    :param val: the string value
    :return: the int or None

    >>> print(str_to_intnone(""))
    None
    >>> print(str_to_intnone("5"))
    5
    """
    return None if len(val) <= 0 else int(val)


def replace_all(find: str, replace: str, src: str) -> str:
    """
    Perform a recursive replacement of strings.

    After applying this function, there will not be any occurence of `find`
    left in `src`. All of them will have been replaced by `replace`. If that
    produces new instances of `find`, these will be replaced as well.
    If `replace` contains `find`, this will lead to an endless loop!

    :param find: the string to find
    :param replace: the string with which it will be replaced
    :param src: the string in which we search
    :return: the string `src`, with all occurrences of find replaced by replace

    >>> replace_all("a", "b", "abc")
    'bbc'
    >>> replace_all("aa", "a", "aaaaa")
    'a'
    >>> replace_all("aba", "a", "abaababa")
    'aa'
    """
    new_len = len(src)
    while True:
        src = src.replace(find, replace)
        old_len = new_len
        new_len = len(src)
        if new_len >= old_len:
            return src


def __replace_double(replace: str, src: str) -> str:
    """
    Replace any double-occurrence of a string with a single occurrence.

    :param replace: the string to replace
    :param src: the source string
    :returns: the updated string
    """
    return replace_all(replace + replace, replace, src)


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


def regex_sub(search: str | Pattern,
              replace: str,
              inside: str) -> str:
    r"""
    Replace all occurrences of 'search' in 'inside' with 'replace'.

    :param search: the regular expression to search
    :param replace: the regular expression to replace it with
    :param inside: the string in which to search/replace
    :return: the new string after the recursive replacement

    >>> regex_sub('[ \t]+\n', '\n', ' bla \nxyz\tabc\t\n')
    ' bla\nxyz\tabc\n'
    >>> regex_sub('[0-9]A', 'X', '23A7AA')
    '2XXA'
    """
    while True:
        text = sub(search, replace, inside, MULTILINE)
        if text is inside:
            return inside
        inside = text
