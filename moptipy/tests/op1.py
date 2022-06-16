"""Functions that can be used to test unary search operators."""
from math import isqrt
from typing import Optional, Callable, Any, Union

from numpy.random import default_rng, Generator

from moptipy.api.operators import Op1, check_op1
from moptipy.api.space import Space
from moptipy.tests.component import validate_component


def validate_op1(op1: Op1,
                 search_space: Space = None,
                 make_search_space_element_valid:
                 Optional[Callable[[Generator, Any], Any]] = lambda _, x: x,
                 number_of_samples: int = 100,
                 min_unique_samples: Union[int, Callable[[int, Space], int]]
                 = lambda samples, space:
                 max(1, min(samples // 2, isqrt(space.n_points())))) -> None:
    """
    Check whether an object is a valid moptipy unary operator.

    :param op1: the operator
    :param search_space: the search space
    :param make_search_space_element_valid: make a point in the search
        space valid
    :param number_of_samples: the number of times to invoke the operator
    :param min_unique_samples: a lambda for computing the number
    :raises ValueError: if `op1` is not a valid `Op1`
    """
    if not isinstance(op1, Op1):
        raise ValueError("Expected to receive an instance of Op1, but "
                         f"got a {type(op1)}.")
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

    if not isinstance(number_of_samples, int):
        raise ValueError(
            f"number_of_samples must be int, but is {number_of_samples}.")
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

    x2 = search_space.create()
    if x2 is None:
        raise ValueError("Space must not return None.")
    if x1 is x2:
        raise ValueError(
            "Search space.create must not return same object instance.")

    if not (hasattr(op1, 'op1') and callable(getattr(op1, 'op1'))):
        raise ValueError("op1 must have method op1.")
    for _ in range(number_of_samples):
        op1.op1(random, x2, x1)
        search_space.validate(x2)
        strstr = search_space.to_str(x2)
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
        raise ValueError(f"expected number of unique values must be int,"
                         f" but is {type(expected)}.")
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
