"""Utilities for multi-objective optimization."""

from typing import TypeAlias, Any, Tuple, List, Union

import numba  # type: ignore
import numpy as np

#: the multi-objective record for problems with only solution space
MORecordX: TypeAlias = Tuple[np.ndarray, Any]
#: the multi-objective record for problems with search and solution space
MORecordXY: TypeAlias = Tuple[np.ndarray, Any, Any]
#: the union of both possible multi-objective archive records
MORecord: TypeAlias = Union[MORecordX, MORecordXY]
#: the type for multi-objective archives
MOArchive: TypeAlias = List[MORecord]


@numba.njit(nogil=True, cache=True)
def domination(a: np.ndarray, b: np.ndarray) -> int:
    """
    Check if one objective vector dominates or is dominated by another one.

    :param a: the first objective vector
    :param b: the second objective value
    :returns: an integer value indicating the domination relationship
    :retval -1: if `a` dominates `b`
    :retval 1: if `b` dominates `a`
    :retval 0: if `a` and `b` are mutually non-dominated, i.e., if neither `a`
        dominates `b` not `b` dominates `a`

    >>> from numpy import array
    >>> domination(array([1, 1, 1]), array([2, 2, 2]))
    -1
    >>> domination(array([1.0, 1.0, 2.0]), array([2.0, 2.0, 1.0]))
    0
    >>> domination(array([2, 2, 2]), array([1, 1, 1]))
    1
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
    return res
