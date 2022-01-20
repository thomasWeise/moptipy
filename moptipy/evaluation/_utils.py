"""Some internal helper functions."""

from math import isfinite, inf, gcd
from typing import Union, Optional, Final

import numba  # type: ignore
import numpy as np

from moptipy.utils import logging


def _ifn_to_str(val: Union[int, float, None]) -> str:
    """
    Convert an integer ot float or `None` to a string.

    :param Union[int, float, None] val: the value
    :return: the string representation
    :rtype: str
    """
    return "" if val is None else logging.num_to_str(val)


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
                         ("." in val) or \
                         ("inf" in val) else int(val)


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


#: The positive limit for doubles that can be represented exactly as ints.
_DBL_INT_LIMIT_P: Final[float] = 9007199254740992.0  # = 1 << 53
#: The negative  limit for doubles that can be represented exactly as ints.
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


def _try_div(a: int, b: int) -> Union[int, float]:
    """
    Try to divide two integers at best precision.

    :param int a: the first integer
    :param int b: the second integer
    :return: a/b
    :rtype: Union[int, float]
    """
    if b == 0:
        return inf
    gc = gcd(a, b)
    a //= gc
    if b == gc:
        return a
    b //= gc
    return _try_int(a / b)


def _try_div2(a: Union[int, float], b: Union[int, float]) \
        -> Union[int, float]:
    """
    Try to divide two numbers at best precision.

    :param int a: the first number
    :param int b: the second number
    :return: a/b
    :rtype: Union[int, float]
    """
    ia = _try_int(a)
    ib = _try_int(b)
    if isinstance(ia, int) and isinstance(ib, int):
        return _try_div(ia, ib)
    return _try_int(ia / ib)


def _check_max_time_millis(max_time_millis: Union[int, float],
                           total_fes: int,
                           total_time_millis: int) -> None:
    """
    Check whether a max-time-millis value is permissible.

    If we set a time limit for a run, then the
    :meth:`~moptipy.api.Process.should_terminate` will become `True`
    approximately after the time limit has expired. However, this also
    could be later (or maybe even earlier) due to the workings of the
    unlderying operating system. And even if
    :meth:`~moptipy.api.Process.should_terminate` if `True`, it is not
    clear whether the optimization algorithm can query it right away.
    Instead, it may be blocked in a long-running objective function
    evaluation or some other computation. Hence, it may actually stop
    later. So we cannot simply require that
    `total_time_millis <= max_time_millis`, as this is not practically
    enforceable. Instead, we will heuristically determine a feasible
    maximum limit for how much longer an algorithm might run.

    :param Union[int, float] max_time_millis: the max time millis threshold
    :param int total_fes: the total FEs performed
    :param int total_time_millis: the measured total time millis
    """
    if total_fes == 1:
        return

    div: Final[float] = (total_fes - 2) if total_fes > 1 else 1
    permitted_limit: Final[float] = \
        60_000 + ((1 + (1.1 * (max_time_millis / div))) * total_fes)
    if total_time_millis > permitted_limit:
        raise ValueError(
            f"If max_time_millis is {max_time_millis} and "
            f"total_fes is {total_fes}, then total_time_millis must "
            f"not be more than {permitted_limit}, but is"
            f"{total_time_millis}.")


@numba.njit(nogil=True)
def _get_reach_index(f: np.ndarray, goal_f: Union[int, float]):
    """
    Compute the offset from the end of `f` when `goal_f` was reached.

    :param np.ndarray f: the raw data array, which must be sorted in
        descending order
    :param Union[int, float] goal_f: the goal f value
    :return: the index, or `0` if `goal_f` was not reached
    :rtype: np.integer

    >>> ft = np.array([10, 9, 8, 5, 3, 2, 1])
    >>> _get_reach_index(ft, 11)
    7
    >>> ft[ft.size - _get_reach_index(ft, 11)]
    10
    >>> _get_reach_index(ft, 10)
    7
    >>> _get_reach_index(ft, 9)
    6
    >>> ft[ft.size - _get_reach_index(ft, 6)]
    5
    >>> _get_reach_index(ft, 1)
    1
    >>> _get_reach_index(ft, 0.9)
    0
    """
    return np.searchsorted(f[::-1], goal_f, side="right")
