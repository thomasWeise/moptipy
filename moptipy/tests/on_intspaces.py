"""Test stuff on integer spaces."""

from typing import Any, Callable, Iterable, cast

import numpy as np
from numpy.random import Generator, default_rng
from pycommons.types import type_error

from moptipy.api.operators import Op0, Op1, Op2
from moptipy.spaces.intspace import IntSpace
from moptipy.tests.op0 import validate_op0
from moptipy.tests.op1 import validate_op1
from moptipy.tests.op2 import validate_op2


def intspaces_for_tests(
        space_filter: Callable[[IntSpace], bool] | None = None)\
        -> Iterable[IntSpace]:
    """
    Get a sequence of integer spaces for tests.

    :param perm_filter: an optional filter to sort out permutations we cannot
        use for testing
    :returns: the sequence of SignedPermutations
    """
    r = default_rng()
    pwrs: list[IntSpace] = [
        IntSpace(1, 0, 1),
        IntSpace(2, 0, 1),
        IntSpace(9, 0, 1),
        IntSpace(1, -1, 1),
        IntSpace(2, -1, 1),
        IntSpace(13, -1, 1),
        IntSpace(1, -12, 17),
        IntSpace(2, -11, 9),
        IntSpace(16, -23, 25),
        IntSpace(1, 0, 127),
        IntSpace(2, -128, 127),
        IntSpace(1, -128, 128),
        IntSpace(1, -2**15, 2**15 - 1),
        IntSpace(1, -2**15, 2**15),
        IntSpace(*map(int, (
            r.integers(1, 100), r.integers(0, 10), r.integers(10, 100)))),
        IntSpace(*map(int, (
            r.integers(1, 100), r.integers(-100, 0), r.integers(1, 100)))),
        IntSpace(*map(int, (
            r.integers(1, 100), r.integers(-100, -10),
            r.integers(-9, 100))))]
    if space_filter is not None:
        if not callable(space_filter):
            raise type_error(space_filter, "perm_filter", None, call=True)
        pwrs = [p for p in pwrs if space_filter(p)]
    r.shuffle(cast("list", pwrs))
    return pwrs


def make_ints_valid(space: IntSpace) -> \
        Callable[[Generator, np.ndarray], np.ndarray]:
    """
    Create a function that can make an int vector valid.

    :param space: the integer space
    :returns: the function
    """
    def __make_valid(prnd: Generator, x: np.ndarray, ppp=space) -> np.ndarray:
        np.copyto(x, prnd.integers(low=ppp.min_value, high=ppp.max_value,
                                   size=ppp.dimension, endpoint=True))
        return x
    return __make_valid


def validate_op0_on_1_intspace(
        op0: Op0 | Callable[[IntSpace], Op0],
        search_space: IntSpace,
        number_of_samples: int | None = None,
        min_unique_samples: int | Callable[[
            int, IntSpace], int] | None = None) -> None:
    """
    Validate the nullary operator on one `IntSpace` instance.

    :param op0: the operator or operator factory
    :param search_space: the search space
    :param number_of_samples: the optional number of samples
    :param min_unique_samples: the optional unique samples
    """
    args: dict[str, Any] = {
        "op0": op0(search_space) if callable(op0) else op0,
        "search_space": search_space,
        "make_search_space_element_valid":
            make_ints_valid(search_space),
    }
    if number_of_samples is not None:
        args["number_of_samples"] = number_of_samples
    if min_unique_samples is not None:
        args["min_unique_samples"] = min_unique_samples
    validate_op0(**args)


def validate_op0_on_intspaces(
        op0: Op0 | Callable[[IntSpace], Op0],
        number_of_samples: int | None = None,
        min_unique_samples: int | Callable[[
            int, IntSpace], int] | None = None,
        space_filter: Callable[[IntSpace], bool] | None = None) \
        -> None:
    """
    Validate the nullary operator on several `IntSpace` instances.

    :param op0: the operator or operator factory
    :param number_of_samples: the optional number of samples
    :param min_unique_samples: the optional unique samples
    :param space_filter: an optional filter to sort out permutations we cannot
        use for testing
    """
    for pwr in intspaces_for_tests(space_filter):
        validate_op0_on_1_intspace(
            op0, pwr, number_of_samples, min_unique_samples)


def validate_op1_on_1_intspace(
        op1: Op1 | Callable[[IntSpace], Op1],
        search_space: IntSpace,
        number_of_samples: int | None = None,
        min_unique_samples: int | Callable[[
            int, IntSpace], int] | None = None) -> None:
    """
    Validate the unary operator on one `IntSpace` instance.

    :param op1: the operator or operator factory
    :param search_space: the search space
    :param number_of_samples: the optional number of samples
    :param min_unique_samples: the optional unique samples
    """
    args: dict[str, Any] = {
        "op1": op1(search_space) if callable(op1) else op1,
        "search_space": search_space,
        "make_search_space_element_valid":
            make_ints_valid(search_space),
    }
    if number_of_samples is not None:
        args["number_of_samples"] = number_of_samples
    if min_unique_samples is not None:
        args["min_unique_samples"] = min_unique_samples
    validate_op1(**args)


def validate_op1_on_intspaces(
        op1: Op1 | Callable[[IntSpace], Op1],
        number_of_samples: int | None = None,
        min_unique_samples: int | Callable[[
            int, IntSpace], int] | None = None,
        space_filter: Callable[[IntSpace], bool] | None = None) \
        -> None:
    """
    Validate the unary operator on several `IntSpace` instances.

    :param op1: the operator or operator factory
    :param number_of_samples: the optional number of samples
    :param min_unique_samples: the optional unique samples
    :param space_filter: an optional filter to sort out permutations we cannot
        use for testing
    """
    for pwr in intspaces_for_tests(space_filter):
        validate_op1_on_1_intspace(
            op1, pwr, number_of_samples, min_unique_samples)


def validate_op2_on_1_intspace(
        op2: Op2 | Callable[[IntSpace], Op2],
        search_space: IntSpace,
        number_of_samples: int | None = None,
        min_unique_samples:
        int | Callable[[int, IntSpace], int] | None
        = None) -> None:
    """
    Validate the binary operator on one `IntSpace` instance.

    :param op2: the operator or operator factory
    :param search_space: the search space
    :param number_of_samples: the optional number of samples
    :param min_unique_samples: the optional unique samples
    """
    args: dict[str, Any] = {
        "op2": op2(search_space) if callable(op2) else op2,
        "search_space": search_space,
        "make_search_space_element_valid": make_ints_valid(search_space),
    }
    if number_of_samples is not None:
        args["number_of_samples"] = number_of_samples
    if min_unique_samples is not None:
        args["min_unique_samples"] = min_unique_samples
    validate_op2(**args)


def validate_op2_on_intspaces(
        op2: Op2 | Callable[[IntSpace], Op2],
        number_of_samples: int | None = None,
        min_unique_samples:
        int | Callable[[int, IntSpace], int] | None = None,
        space_filter: Callable[[IntSpace], bool] | None = None) -> None:
    """
    Validate the binary operator on several `IntSpace` instances.

    :param op2: the operator or operator factory
    :param number_of_samples: the optional number of samples
    :param min_unique_samples: the optional unique samples
    :param space_filter: an optional filter to sort out permutations we cannot
        use for testing
    """
    for bst in intspaces_for_tests(space_filter):
        validate_op2_on_1_intspace(
            op2, bst, number_of_samples, min_unique_samples)
