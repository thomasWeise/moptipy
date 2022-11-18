"""Functions that can be used to test binary search operators."""
from math import isqrt
from typing import Any, Callable

from numpy.random import Generator, default_rng

from moptipy.api.operators import Op2, check_op2
from moptipy.api.space import Space
from moptipy.tests.component import validate_component
from moptipy.utils.types import type_error


def validate_op2(op2: Op2,
                 search_space: Space = None,
                 make_search_space_element_valid:
                 Callable[[Generator, Any], Any] | None = lambda _, x: x,
                 number_of_samples: int = 100,
                 min_unique_samples: int | Callable[[int, Space], int]
                 = lambda samples, space:
                 max(1, min(samples // 2, isqrt(space.n_points())))) -> None:
    """
    Check whether an object is valid a moptipy binary operator.

    :param op2: the operator
    :param search_space: the search space
    :param make_search_space_element_valid: make a point in the search
        space valid
    :param number_of_samples: the number of times to invoke the operator
    :param min_unique_samples: a lambda for computing the number
    :raises ValueError: if `op2` is not a valid instance of
        :class:`~moptipy.api.operators.Op2`
    :raises TypeError: if incorrect types are encountered
    """
    if not isinstance(op2, Op2):
        raise type_error(op2, "op2", Op2)
    if op2.__class__ == Op2:
        raise ValueError("Cannot use abstract base Op2 directly.")
    check_op2(op2)
    validate_component(op2)

    count: int = 0
    if search_space is not None:
        count += 1
    if make_search_space_element_valid is not None:
        count += 1
    if count <= 0:
        return
    if count < 2:
        raise ValueError(
            "either provide both of search_space and "
            "make_search_space_element_valid or none.")
    if not isinstance(number_of_samples, int):
        raise type_error(number_of_samples, "number_of_samples", int)
    if not (1 <= number_of_samples <= 1_000_000):
        raise ValueError("number_of_samples must be in 1..1_000_000, "
                         f"but is {number_of_samples}.")

    random = default_rng()

    x1 = search_space.create()
    if x1 is None:
        raise ValueError("Space must not return None.")
    x1 = make_search_space_element_valid(random, x1)
    if x1 is None:
        raise ValueError("validator turned point to None?")
    search_space.validate(x1)

    seen = set()

    strstr = search_space.to_str(x1)
    if (not isinstance(strstr, str)) or (len(strstr) <= 0):
        raise ValueError("to_str produces either no string or "
                         f"empty string, namely {strstr}.")
    seen.add(strstr)
    x2: Any = None
    for _ in range(1000):
        x2 = search_space.create()
        if x2 is None:
            raise ValueError("Space must not return None.")
        x2 = make_search_space_element_valid(random, x2)
        if x2 is None:
            raise ValueError("validator turned point to None?")
        if x1 is x2:
            raise ValueError(
                "Search space.create must not return same object instance.")
        search_space.validate(x2)
        strstr = search_space.to_str(x2)
        if (not isinstance(strstr, str)) or (len(strstr) <= 0):
            raise ValueError("to_str produces either no string or "
                             f"empty string, namely {strstr}.")
        seen.add(strstr)
        if len(seen) > 1:
            break
    if len(seen) <= 1:
        raise ValueError("failed to create two different initial elements.")

    x3 = search_space.create()
    if x3 is None:
        raise ValueError("Space must not return None.")
    if (x1 is x3) or (x2 is x3):
        raise ValueError(
            "Search space.create must not return same object instance.")

    if not (hasattr(op2, "op2") and callable(getattr(op2, "op2"))):
        raise ValueError("op2 must have method op2.")
    for _ in range(number_of_samples):
        op2.op2(random, x3, x2, x1)
        search_space.validate(x3)
        strstr = search_space.to_str(x3)
        if (not isinstance(strstr, str)) or (len(strstr) <= 0):
            raise ValueError("to_str produces either no string or "
                             f"empty string, namely '{strstr}'.")
        seen.add(strstr)

    expected: int
    if callable(min_unique_samples):
        expected = min_unique_samples(number_of_samples, search_space)
    else:
        expected = min_unique_samples
    if not isinstance(expected, int):
        raise type_error(expected, "expected", int)
    if expected > number_of_samples:
        raise ValueError(
            f"number of expected unique samples {expected} cannot be larger "
            f"than total number of samples {number_of_samples}.")
    if expected <= 0:
        raise ValueError(f"expected number of unique values must be positive,"
                         f" but is {expected}.")
    if len(seen) < expected:
        raise ValueError(
            f"It is expected that at least {expected} different elements "
            "will be created by unary search operator from "
            f"{number_of_samples} samples, but we only "
            f"got {len(seen)} different points.")
