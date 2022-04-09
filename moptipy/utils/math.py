"""Simple routines for handling numbers and numerical stuff."""

from math import isfinite, inf, gcd
from typing import Union, Final

from moptipy.utils.types import type_error

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
    raise type_error(val, "val", (int, float))


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
