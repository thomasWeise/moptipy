"""Routines for handling strings."""

import math
from typing import Union, Optional

from moptipy.utils.math import __try_int


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
    if math.isnan(x):
        raise ValueError(f"'{s}' not permitted.")
    if s.endswith(".0"):
        return s[:-2]
    return s


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
