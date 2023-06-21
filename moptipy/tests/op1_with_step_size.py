"""Functions for testing unary search operators with step size."""
from math import isfinite
from typing import Any, Callable, Iterable

from numpy.random import Generator, default_rng

from moptipy.api.operators import Op1WithStepSize, check_op1_with_step_size
from moptipy.api.space import Space
from moptipy.tests.op1 import default_min_unique_samples, validate_op1
from moptipy.utils.types import type_error


def validate_op1_with_step_size(
        op1: Op1WithStepSize,
        search_space: Space | None = None,
        make_search_space_element_valid:
        Callable[[Generator, Any], Any] | None = lambda _, x: x,
        number_of_samples: int = 100,
        min_unique_samples: int | Callable[[int, Space], int]
        = default_min_unique_samples,
        step_sizes: Iterable[float] = (),
        get_step_size: Callable[[
            Space, Any, Any], float | None] | None = None) -> None:
    """
    Check whether an object is a valid moptipy unary operator with step size.

    :param op1: the operator
    :param search_space: the search space
    :param make_search_space_element_valid: make a point in the search
        space valid
    :param number_of_samples: the number of times to invoke the operator
    :param min_unique_samples: a lambda for computing the number
    :param step_sizes: the step sizes to test
    :param get_step_size: try to get the step size difference from two space
        elements
    :raises ValueError: if `op1` is not a valid instance of
        :class:`~moptipy.api.operators.Op1`
    :raises TypeError: if incorrect types are encountered
    """
    if not isinstance(op1, Op1WithStepSize):
        raise type_error(op1, "op1", Op1WithStepSize)
    if op1.__class__ == Op1WithStepSize:
        raise ValueError("Cannot use abstract base Op1WithStepSize directly.")
    check_op1_with_step_size(op1)
    if not isinstance(step_sizes, Iterable):
        raise type_error(step_sizes, "step_sizes", Iterable)
    if (get_step_size is not None) and (not callable(get_step_size)):
        raise type_error(get_step_size, "get_step_size", None, call=True)

    validate_op1(op1, search_space, make_search_space_element_valid,
                 number_of_samples, min_unique_samples)

    random: Generator | None = None
    x: Any = None
    x_copy: Any = None
    dest: Any = None
    if search_space is not None:
        if random is None:
            random = default_rng()
        x = search_space.create()
        if x is None:
            raise ValueError(
                f"search_space.create()=None for {search_space}")
        x_copy = search_space.create()
        if x_copy is None:
            raise ValueError(
                f"search_space.create()=None for {search_space}")
        dest = search_space.create()
        if dest is None:
            raise ValueError(
                f"search_space.create()=None for {search_space}")

    for i, step_size in enumerate(step_sizes):
        if not isinstance(step_size, float):
            raise type_error(step_size, f"step_sizes[{i}]", float)
        if not (isfinite(step_size) and (0 <= step_size <= 1)):
            raise ValueError(f"Forbidden step_sizes[{i}]={step_size}.")
        if (random is not None) and (search_space is not None):
            if make_search_space_element_valid is not None:
                x = make_search_space_element_valid(random, x)
            if x is None:
                raise ValueError(
                    "make_search_space_element_valid(search_space.create())="
                    f"None for {search_space}")
            search_space.validate(x)
            search_space.copy(x_copy, x)
            if not search_space.is_equal(x_copy, x):
                raise ValueError(
                    f"error when copying {x}, got {x_copy} on {search_space}")
            op1.op1(random, dest, x, step_size)
            search_space.validate(dest)
            if search_space.is_equal(dest, x_copy):
                raise ValueError(
                    f"operator copies source for step_size={step_size} "
                    f"on {search_space} and {dest}")
            if not search_space.is_equal(x, x_copy):
                raise ValueError(
                    f"operator modifies source for step_size={step_size} "
                    f"on {search_space}")
            if get_step_size is not None:
                found_step_size: float | None = \
                    get_step_size(search_space, x, dest)
                if found_step_size is None:
                    continue
                if not (isfinite(found_step_size)
                        and (0 <= found_step_size <= 1)):
                    raise ValueError(
                        f"invalid detected step size {found_step_size} "
                        f"for {search_space}.")
                if found_step_size != step_size:
                    raise ValueError(
                        f"step_size={step_size} but {op1} actually "
                        f"performs step of size {found_step_size} for "
                        f"x={x} and returns {dest} for {search_space}.")
