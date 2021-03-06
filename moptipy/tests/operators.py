"""Functions that can be used to test search operators."""
from math import isqrt
from typing import Optional, Callable

from numpy.random import default_rng

from moptipy.api.operators import Op0, Op1
from moptipy.api.space import Space
from moptipy.tests.component import check_component


def check_op0(op0: Op0,
              space: Space = None,
              make_valid: Optional[Callable] = lambda x: x) -> None:
    """
    Check whether an object is a moptipy nullary operator.

    :param op0: the operator
    :param space: the space
    :param make_valid: make a point in the search space valid
    :raises ValueError: if `op0` is not a valid `Op0Shuffle`
    """
    if not isinstance(op0, Op0):
        raise ValueError("Expected to receive an instance of Op0Shuffle, but "
                         f"got a {type(op0)}.")
    check_component(component=op0)

    if (space is None) or (make_valid is None):
        return

    x = space.create()
    if x is None:
        raise ValueError("Space must not return None.")

    x = make_valid(x)
    if x is None:
        raise ValueError("Make valid failed.")
    space.validate(x)

    random = default_rng()
    seen = set()
    max_count = 100

    for _ in range(max_count):
        op0.op0(random, x)
        space.validate(x)
        strstr = space.to_str(x)
        if (not isinstance(strstr, str)) or (len(strstr) <= 0):
            raise ValueError("to_str produces either no string or "
                             f"empty string, namely '{strstr}'.")
        seen.add(strstr)

    expected = max(1, min(max_count // 2, isqrt(space.scale())))
    if len(seen) < expected:
        raise ValueError(
            f"It is expected that at least {expected} different elements "
            "will be created by nullary search operator from "
            f"{max_count} samples, but we only got {len(seen)} "
            "different points.")


def check_op1(op1: Op1,
              space: Space = None,
              make_valid: Optional[Callable] = lambda x: x) -> None:
    """
    Check whether an object is a moptipy unary operator.

    :param op1: the operator
    :param space: the space
    :param make_valid: a method that can turn a point from the
        space into a valid point
    :raises ValueError: if `op1` is not a valid `Op1`
    """
    if not isinstance(op1, Op1):
        raise ValueError("Expected to receive an instance of Op1, but "
                         f"got a {type(op1)}.")
    check_component(component=op1)

    if (space is None) or (make_valid is None):
        return

    x1 = space.create()
    if x1 is None:
        raise ValueError("Space must not return None.")
    x1 = make_valid(x1)
    if x1 is None:
        raise ValueError("validator turned point to None?")
    space.validate(x1)

    random = default_rng()
    seen = set()
    max_count = 100

    strstr = space.to_str(x1)
    if (not isinstance(strstr, str)) or (len(strstr) <= 0):
        raise ValueError("to_str produces either no string or "
                         f"empty string, namely {strstr}.")
    seen.add(strstr)

    x2 = space.create()
    if x2 is None:
        raise ValueError("Space must not return None.")

    for _ in range(max_count):
        op1.op1(random, x1, x2)
        space.validate(x2)
        strstr = space.to_str(x2)
        if (not isinstance(strstr, str)) or (len(strstr) <= 0):
            raise ValueError("to_str produces either no string or "
                             f"empty string, namely '{strstr}'.")
        seen.add(strstr)

    expected = max(2, min(3, isqrt(space.scale())))
    if len(seen) < expected:
        raise ValueError(
            f"It is expected that at least {expected} different elements "
            "will be created by unary search operator from "
            f"{max_count} samples, but we only got {len(seen)} different "
            "points.")
