"""Some basic type conversation routines."""

import math
from math import isfinite, inf, gcd
from typing import Union, Optional, Final


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

    :param Union[int, float, None] val: the value
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


#: The positive limit for doubles that can be represented exactly as ints.
DBL_INT_LIMIT_P: Final[float] = 9007199254740992.0  # = 1 << 53
#: The negative  limit for doubles that can be represented exactly as ints.
_DBL_INT_LIMIT_N: Final[float] = -DBL_INT_LIMIT_P


def __try_int(val: float) -> Union[int, float]:
    """
    Convert a float to an int without any fancy checks.

    :param val: the flot
    :returns: the float or int
    """
    if _DBL_INT_LIMIT_N <= val <= DBL_INT_LIMIT_P:
        a = int(val)
        if a == val:
            return a
    return val


def try_int(val: Union[int, float]) -> Union[int, float]:
    """
    Attempt to convert a float to an integer.

    :param val: the input value
    :return: an `int` if `val` can be represented as `int` without loss of
        precision, `val` otherwise

    >>> print(type(try_int(10.5)))
    <class 'float'>
    >>> print(type(try_int(10)))
    <class 'int'>
    """
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        if not isfinite(val):
            raise ValueError(f"Value must be finite, but is {val}.")
        if _DBL_INT_LIMIT_N <= val <= DBL_INT_LIMIT_P:
            a = int(val)
            if a == val:
                return a
        return val
    raise TypeError(f"Value must be int or float, but is {type(val)}.")


def try_int_div(a: int, b: int) -> Union[int, float]:
    """
    Try to divide two integers at best precision.

    :param a: the first integer
    :param b: the second integer
    :return: a/b

    >>> print(type(try_int_div(10, 2)))
    <class 'int'>
    >>> print(type(try_int_div(10, 3)))
    <class 'float'>
    """
    if b == 0:
        return inf
    gc: Final[int] = gcd(a, b)
    a //= gc
    if b == gc:
        return a
    b //= gc
    val: Final[float] = a / b
    if not isfinite(val):
        raise ValueError(f"Value must be finite, but is {a}/{b}={val}.")
    return __try_int(val)


def try_float_div(a: Union[int, float], b: Union[int, float]) \
        -> Union[int, float]:
    """
    Try to divide two numbers at best precision.

    :param a: the first number
    :param b: the second number
    :return: a/b

    >>> print(type(try_float_div(10, 2)))
    <class 'int'>
    >>> print(type(try_float_div(10, 3)))
    <class 'float'>
    >>> print(type(try_float_div(10, 0.5)))
    <class 'int'>
    """
    ia = try_int(a)
    ib = try_int(b)
    if isinstance(ia, int) and isinstance(ib, int):
        return try_int_div(ia, ib)
    val: Final[float] = ia / ib
    if not isfinite(val):
        raise ValueError(f"Value must be finite, but is {a}/{b}={val}.")
    return __try_int(val)


def classname(obj) -> str:
    """
    Get the fully-qualified class name of an object.

    :param obj: the object
    :returns: the fully-qualified class name of the object

    >>> from moptipy.examples.jssp.instance import Instance
    >>> print(classname(Instance.from_resource("demo")))
    moptipy.examples.jssp.instance.Instance
    >>> from numpy.random import default_rng
    >>> print(classname(default_rng()))
    numpy.random._generator.Generator
    """
    if obj is None:
        return "None"

    candidate1 = ""
    if hasattr(obj, "__class__"):
        cc = obj.__class__
        if hasattr(cc, "__qualname__"):
            candidate1 = cc.__qualname__
            if hasattr(obj, "__module__"):
                module = obj.__module__
                if module is not None:
                    candidate1 = f"{module}.{candidate1}"

    candidate2 = str(type(obj))
    if candidate2.startswith("<class '"):
        candidate2 = candidate2[8:-2]

    return candidate1 if len(candidate1) > len(candidate2) else candidate2
