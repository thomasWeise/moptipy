"""Functions that can be used to test nullary search operators."""
from math import isqrt
from typing import Optional, Callable, Any, Union

from numpy.random import default_rng, Generator

from moptipy.api.operators import Op0, check_op0
from moptipy.api.space import Space
from moptipy.tests.component import validate_component
from moptipy.utils.types import type_error


def validate_op0(op0: Op0,
                 search_space: Space = None,
                 make_search_space_element_valid:
                 Optional[Callable[[Generator, Any], Any]] = lambda _, x: x,
                 number_of_samples: int = 100,
                 min_unique_samples: Union[int, Callable[[int, Space], int]]
                 = lambda samples, space:
                 max(1, min(samples // 2, isqrt(space.n_points())))) -> None:
    """
    Check whether an object is a valid moptipy nullary operator.

    :param op0: the operator
    :param search_space: the search space
    :param make_search_space_element_valid: make a point in the search
        space valid
    :param number_of_samples: the number of times to invoke the operator
    :param min_unique_samples: a lambda for computing the number
    :raises ValueError: if `op0` is not a valid instance of
        :class:`~moptipy.api.operators.Op0`
    :raises TypeError: if invalid types are encountered
    """
    if not isinstance(op0, Op0):
        raise type_error(op0, "op0", Op0)
    if op0.__class__ == Op0:
        raise ValueError("Cannot use abstract base Op0 directly.")
    check_op0(op0)
    validate_component(op0)

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
    x = search_space.create()
    if x is None:
        raise ValueError("Space must not return None.")
    x = make_search_space_element_valid(random, x)
    if x is None:
        raise ValueError("Make valid failed.")
    search_space.validate(x)

    seen = set()

    if not (hasattr(op0, 'op0') and callable(getattr(op0, 'op0'))):
        raise ValueError("op0 must have method op0.")
    for _ in range(number_of_samples):
        op0.op0(random, x)
        search_space.validate(x)
        strstr = search_space.to_str(x)
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
            "will be created by nullary search operator from "
            f"{number_of_samples} samples, but we only got {len(seen)} "
            "different points.")
