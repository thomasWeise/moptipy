"""Some internal helper functions."""

from math import isfinite
from typing import Union, Optional, Final

from moptipy.utils import logging


def _if_to_str(val: Union[int, float]) -> str:
    """
    Convert an integer ot float to a string.

    :param Union[int, float] val: the value
    :return: the string representation
    :rtype: str
    """
    return str(val) if isinstance(val, int) \
        else logging.float_to_str(val)


def _ifn_to_str(val: Union[int, float, None]) -> str:
    """
    Convert an integer ot float or `None` to a string.

    :param Union[int, float, None] val: the value
    :return: the string representation
    :rtype: str
    """
    return "" if val is None else \
        str(val) if isinstance(val, int) \
        else logging.float_to_str(val)


def _in_to_str(val: Optional[int]) -> str:
    """
    Convert an integer or `None` to a string.

    :param Optional[int] val: the value
    :return: the string representation
    :rtype: str
    """
    return "" if val is None else str(val)


def _str_to_if(val: str) -> Union[int, float]:
    """
    Convert a string to an int or float.

    :param str val: the string value
    :return: the int or float
    :rtype: Union[int, float]
    """
    return float(val) if ("e" in val) or \
                         ("E" in val) or \
                         ("." in val) else int(val)


def _str_to_ifn(val: str) -> Union[int, float, None]:
    """
    Convert a string to an int or float or None.

    :param str val: the string value
    :return: the int or float or None
    :rtype: Union[int, float, None]
    """
    return None if len(val) <= 0 else _str_to_if(val)


def _str_to_in(val: str) -> Optional[int]:
    """
    Convert a string to an int or None.

    :param str val: the string value
    :return: the int or None
    :rtype: Optional[int, None]
    """
    return None if len(val) <= 0 else int(val)


#: The positive limit for doubles that can be represent exactly as ints.
_DBL_INT_LIMIT_P: Final[float] = 9007199254740992.0  # = 1 << 53
#: The negative  limit for doubles that can be represent exactly as ints.
_DBL_INT_LIMIT_N: Final[float] = -_DBL_INT_LIMIT_P


def _try_int(val: Union[int, float]) -> Union[int, float]:
    """
    Attempt to convert a float to an integer.

    :param Union[int, float] val: the input value
    :return: an `int` if `val` can be represented as `int` without loss of
        precision, `val` otherwise
    :rtype: Union[int, float]
    """
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        if not isfinite(val):
            raise ValueError(f"Value must be finite, but is {val}.")
        if _DBL_INT_LIMIT_N <= val <= _DBL_INT_LIMIT_P:
            a = int(val)
            if a == val:
                return a
            return val
    raise TypeError(f"Value must be int or float, but is {type(val)}.")
