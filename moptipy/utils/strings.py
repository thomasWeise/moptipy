"""Routines for handling strings."""

from math import isnan
from re import compile as _compile, sub, MULTILINE
from typing import Union, Optional, Final, Iterable, Pattern

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
    """
    if x == 0:
        return "0"
    s = repr(x)
    if isnan(x):
        raise ValueError(f"'{s}' not permitted.")
    if s.endswith(".0"):
        return s[:-2]
    return s


def num_to_str_for_name(x: Union[int, float]) -> str:
    """
    Convert a float to a string for use in a component name.

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
    """
    return num_to_str(x).replace('.', DECIMAL_DOT_REPLACEMENT)\
        .replace('-', MINUS_REPLACEMENT)


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
    return 'T' if value else 'F'


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
    raise ValueError(f"Expected 'T' or 'F', but got '{value}'.")


def num_to_str(value: Union[int, float]) -> str:
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


def intfloatnone_to_str(val: Union[int, float, None]) -> str:
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


def intnone_to_str(val: Optional[int]) -> str:
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


def str_to_intfloat(val: str) -> Union[int, float]:
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


def str_to_intfloatnone(val: str) -> Union[int, float, None]:
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


def str_to_intnone(val: str) -> Optional[int]:
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

    name = sub(__PATTERN_SPACE_BEFORE_MINUS, '', name)
    name = sub(__PATTERN_MULTIPLE_WHITESPACE, PART_SEPARATOR, name)
    name = __replace_double("_", name)

    if name.startswith("_"):
        name = name[1:]

    if name.endswith("_"):
        name = name[:len(name) - 1]

    if len(name) <= 0:
        raise ValueError(
            f"Sanitized name must not become empty, but '{orig_name}' does.")

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


def regex_sub(search: Union[str, Pattern],
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
