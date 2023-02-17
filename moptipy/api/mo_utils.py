"""Utilities for multi-objective optimization."""

import numba  # type: ignore
import numpy as np


@numba.njit(cache=True)
def dominates(a: np.ndarray, b: np.ndarray) -> int:
    """
    Check if one objective vector dominates or is dominated by another one.

    :param a: the first objective vector
    :param b: the second objective value
    :returns: an integer value indicating the domination relationship
    :retval -1: if `a` dominates `b`
    :retval 1: if `b` dominates `a`
    :retval 2: if `b` equals `a`
    :retval 0: if `a` and `b` are mutually non-dominated, i.e., if neither `a`
        dominates `b` not `b` dominates `a` and `b` is also different from `a`

    >>> from numpy import array
    >>> dominates(array([1, 1, 1]), array([2, 2, 2]))
    -1
    >>> dominates(array([1.0, 1.0, 2.0]), array([2.0, 2.0, 1.0]))
    0
    >>> dominates(array([2, 2, 2]), array([1, 1, 1]))
    1
    >>> dominates(array([2, 2, 2]), array([2, 2, 2]))
    2
    """
    res: int = 0
    for i, av in enumerate(a):
        bv = b[i]
        if av < bv:
            if res == 0:
                res = -1
            elif res == 1:
                return 0
        elif bv < av:
            if res == 0:
                res = 1
            elif res == -1:
                return 0
    return 2 if res == 0 else res


@numba.njit(cache=True)
def lexicographic(a: np.ndarray, b: np.ndarray) -> int:
    """
    Compare two arrays lexicographically.

    :param a: the first objective vector
    :param b: the second objective value
    :returns: `-1` if `a` is lexicographically less than `b`, `1` if `b` is
        less than `a`, `0` otherwise
    :retval -1: if `a` is lexicographically less than `b`
    :retval 1: if `b` is lexicographically less than `a`
    :retval 2: if `b` equals `a`

    >>> from numpy import array
    >>> lexicographic(array([1, 1, 1]), array([2, 2, 2]))
    -1
    >>> lexicographic(array([1, 1, 1]), array([1, 1, 2]))
    -1
    >>> lexicographic(array([2, 2, 2]), array([1, 1, 1]))
    1
    >>> lexicographic(array([2, 2, 2]), array([2, 2, 1]))
    1
    >>> lexicographic(array([2, 2, 2]), array([2, 2, 2]))
    0
    """
    for i, f in enumerate(a):
        k = b[i]
        if f < k:
            return -1
        if f > k:
            return 1
    return 0
