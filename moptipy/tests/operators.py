from numpy.random import default_rng

from moptipy.api.operators import Op0, Op1
from moptipy.api.space import Space
from moptipy.tests.component import check_component
from math import isqrt
from typing import Optional, Callable


def check_op0(op0: Op0 = None,
              space: Space = None):
    """
    Check whether an object is a moptipy nullary operator.
    :param op0: the operator
    :param space: the space
    :raises ValueError: if `op0` is not a valid `Op0`
    """
    if not isinstance(op0, Op0):
        raise ValueError("Expected to receive an instance of Op0, but "
                         "got a '" + str(type(op0)) + "'.")
    check_component(component=op0)

    if not (space is None):
        random = default_rng()

        seen = set()
        max_count = 100

        x = space.create()
        if x is None:
            raise ValueError("Space must not return None.")

        for i in range(max_count):
            op0.op0(random, x)
            space.validate(x)
            strstr = space.to_str(x)
            if (not isinstance(strstr, str)) or (len(str) <= 0):
                raise ValueError("to_str produces either no string or "
                                 "empty string, namely '" + str(strstr) + "'.")
            seen.add(strstr)

        expected = max(1, min(max_count // 2, isqrt(space.scale())))
        if len(seen) < expected:
            raise ValueError("It is expected that at least " + str(expected)
                             + " different elements will be created by "
                               "nullary search operator from "
                             + str(max_count) + " samples, but we only got "
                             + str(len(seen)) + " different points.")


def check_op1(op1: Op1 = None,
              space: Space = None,
              make_search_space_valid: Optional[Callable] = lambda x: x):
    """
    Check whether an object is a moptipy unary operator.
    :param op1: the operator
    :param space: the space
    :param make_search_space_valid: a method that can turn a point from the
    space into a valid point
    :raises ValueError: if `op1` is not a valid `Op1`
    """
    if not isinstance(op1, Op1):
        raise ValueError("Expected to receive an instance of Op1, but "
                         "got a '" + str(type(op1)) + "'.")
    check_component(component=op1)

    if not ((space is None) or (make_search_space_valid is None)):
        random = default_rng()

        seen = set()
        max_count = 100

        x1 = space.create()
        if x1 is None:
            raise ValueError("Space must not return None.")
        x1 = make_search_space_valid(x1)
        if x1 is None:
            raise ValueError("validator turned point to None?")
        space.validate(x1)
        strstr = space.to_str(x1)
        if (not isinstance(strstr, str)) or (len(str) <= 0):
            raise ValueError("to_str produces either no string or "
                             "empty string, namely '" + str(strstr) + "'.")
        seen.add(strstr)

        x2 = space.create()
        if x2 is None:
            raise ValueError("Space must not return None.")

        for i in range(max_count):
            op1.op1(random, x1, x2)
            space.validate(x2)
            strstr = space.to_str(x2)
            if (not isinstance(strstr, str)) or (len(str) <= 0):
                raise ValueError("to_str produces either no string or "
                                 "empty string, namely '" + str(strstr) + "'.")
            seen.add(strstr)

        expected = max(2, min(3, isqrt(space.scale())))
        if len(seen) < expected:
            raise ValueError("It is expected that at least " + str(expected)
                             + " different elements will be created by "
                               "unary search operator from "
                             + str(max_count) + " samples, but we only got "
                             + str(len(seen)) + " different points.")
