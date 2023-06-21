"""Functions that can be used to test unary search operators."""
from math import isqrt
from typing import Any, Callable, Final

from numpy.random import Generator, default_rng

from moptipy.api.operators import Op1, check_op1
from moptipy.api.space import Space
from moptipy.tests.component import validate_component
from moptipy.utils.types import check_int_range, type_error


def default_min_unique_samples(samples: int, space: Space) -> int:
    """
    Compute the default number of minimum unique samples.

    :param samples: the number of samples
    :param space: the space
    :returns: the number of samples
    """
    return max(1, min(samples // 2, isqrt(space.n_points())))


def validate_op1(op1: Op1,
                 search_space: Space | None = None,
                 make_search_space_element_valid:
                 Callable[[Generator, Any], Any] | None = lambda _, x: x,
                 number_of_samples: int = 100,
                 min_unique_samples: int | Callable[[int, Space], int]
                 = default_min_unique_samples) -> None:
    """
    Check whether an object is a valid moptipy unary operator.

    :param op1: the operator
    :param search_space: the search space
    :param make_search_space_element_valid: make a point in the search
        space valid
    :param number_of_samples: the number of times to invoke the operator
    :param min_unique_samples: a lambda for computing the number
    :raises ValueError: if `op1` is not a valid instance of
        :class:`~moptipy.api.operators.Op1`
    :raises TypeError: if incorrect types are encountered
    """
    if not isinstance(op1, Op1):
        raise type_error(op1, "op1", Op1)
    if op1.__class__ == Op1:
        raise ValueError("Cannot use abstract base Op1 directly.")
    check_op1(op1)
    validate_component(op1)

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
    check_int_range(number_of_samples, "number_of_samples", 1, 1_000_000)
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

    x2 = search_space.create()
    if x2 is None:
        raise ValueError("Space must not return None.")
    if x1 is x2:
        raise ValueError(
            "Search space.create must not return same object instance.")

    if not (hasattr(op1, "op1") and callable(getattr(op1, "op1"))):
        raise ValueError("op1 must have method op1.")
    for _ in range(number_of_samples):
        op1.op1(random, x2, x1)
        search_space.validate(x2)
        strstr = search_space.to_str(x2)
        if (not isinstance(strstr, str)) or (len(strstr) <= 0):
            raise ValueError("to_str produces either no string or "
                             f"empty string, namely {strstr!r}.")
        seen.add(strstr)

    expected: Final[int] = check_int_range(min_unique_samples(
        number_of_samples, search_space) if callable(
        min_unique_samples) else min_unique_samples,
        "expected", 1, number_of_samples)
    if len(seen) < expected:
        raise ValueError(
            f"It is expected that at least {expected} different elements "
            "will be created by unary search operator from "
            f"{number_of_samples} samples, but we only "
            f"got {len(seen)} different points.")
