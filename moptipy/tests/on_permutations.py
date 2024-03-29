"""Test stuff on permutations with repetitions."""

from typing import Any, Callable, Iterable, cast

import numpy as np
from numpy.random import Generator, default_rng
from pycommons.types import type_error

from moptipy.api.operators import Op0, Op1, Op1WithStepSize, Op2
from moptipy.spaces.permutations import Permutations
from moptipy.tests.op0 import validate_op0
from moptipy.tests.op1 import validate_op1
from moptipy.tests.op1_with_step_size import validate_op1_with_step_size
from moptipy.tests.op2 import validate_op2


def permutations_for_tests(
        perm_filter: Callable[[Permutations], bool] | None = None) \
        -> Iterable[Permutations]:
    """
    Get a sequence of permutations for tests.

    :param perm_filter: an optional filter to sort out permutations we cannot
        use for testing
    :returns: the sequence of Permutations
    """
    r = default_rng()
    pwrs: list[Permutations] = [
        Permutations.standard(2),
        Permutations.standard(3),
        Permutations.standard(4),
        Permutations.standard(5),
        Permutations.standard(6),
        Permutations.standard(12),
        Permutations.standard(23),
        Permutations.with_repetitions(2, 2),
        Permutations.with_repetitions(2, 3),
        Permutations.with_repetitions(3, 2),
        Permutations.with_repetitions(3, 3),
        Permutations.with_repetitions(5, 5),
        Permutations.with_repetitions(int(r.integers(6, 10)),
                                      int(r.integers(2, 7))),
        Permutations.with_repetitions(int(r.integers(2, 5)),
                                      int(r.integers(6, 10))),
        Permutations.with_repetitions(int(r.integers(130, 500)),
                                      int(r.integers(2, 200))),
        Permutations([1, 1, 1, 1, 1, 5, 5, 3]),
        Permutations([0, 1, 1, 1, 1, 1])]
    if perm_filter is not None:
        if not callable(perm_filter):
            raise type_error(perm_filter, "perm_filter", None, call=True)
        pwrs = [p for p in pwrs if perm_filter(p)]
    r.shuffle(cast(list, pwrs))
    return pwrs


def make_permutation_valid(pwr: Permutations) -> \
        Callable[[Generator, np.ndarray], np.ndarray]:
    """
    Create a function that can make permutations with repetitions valid.

    :param pwr: the permutations
    :returns: the function
    """
    def __make_valid(prnd: Generator, x: np.ndarray, ppp=pwr) -> np.ndarray:
        np.copyto(x, ppp.blueprint)
        prnd.shuffle(x)
        return x
    return __make_valid


def validate_op0_on_1_permutations(
        op0: Op0 | Callable[[Permutations], Op0],
        search_space: Permutations,
        number_of_samples: int | None = None,
        min_unique_samples: int | Callable[[
            int, Permutations], int] | None = None) -> None:
    """
    Validate the nullary operator on one `Permutations` instance.

    :param op0: the operator or operator factory
    :param search_space: the search space
    :param number_of_samples: the optional number of samples
    :param min_unique_samples: the optional unique samples
    """
    args: dict[str, Any] = {
        "op0": op0(search_space) if callable(op0) else op0,
        "search_space": search_space,
        "make_search_space_element_valid":
            make_permutation_valid(search_space),
    }
    if number_of_samples is not None:
        args["number_of_samples"] = number_of_samples
    if min_unique_samples is not None:
        args["min_unique_samples"] = min_unique_samples
    validate_op0(**args)


def validate_op0_on_permutations(
        op0: Op0 | Callable[[Permutations], Op0],
        number_of_samples: int | None = None,
        min_unique_samples: int | Callable[[
            int, Permutations], int] | None = None,
        perm_filter: Callable[[Permutations], bool] | None = None) -> None:
    """
    Validate the nullary operator on several `Permutations` instances.

    :param op0: the operator or operator factory
    :param number_of_samples: the optional number of samples
    :param min_unique_samples: the optional unique samples
    :param perm_filter: an optional filter to sort out permutations we cannot
        use for testing
    """
    for pwr in permutations_for_tests(perm_filter):
        validate_op0_on_1_permutations(op0, pwr, number_of_samples,
                                       min_unique_samples)


def validate_op1_on_1_permutations(
        op1: Op1 | Callable[[Permutations], Op1],
        search_space: Permutations,
        number_of_samples: int | None = None,
        min_unique_samples: int | Callable[[
            int, Permutations], int] | None = None) -> None:
    """
    Validate the unary operator on one `Permutations` instance.

    :param op1: the operator or operator factory
    :param search_space: the search space
    :param number_of_samples: the optional number of samples
    :param min_unique_samples: the optional unique samples
    """
    args: dict[str, Any] = {
        "op1": op1(search_space) if callable(op1) else op1,
        "search_space": search_space,
        "make_search_space_element_valid":
            make_permutation_valid(search_space),
    }
    if number_of_samples is not None:
        args["number_of_samples"] = number_of_samples
    if min_unique_samples is not None:
        args["min_unique_samples"] = min_unique_samples
    validate_op1(**args)


def validate_op1_on_permutations(
        op1: Op1 | Callable[[Permutations], Op1],
        number_of_samples: int | None = None,
        min_unique_samples: int | Callable[[
            int, Permutations], int] | None = None,
        perm_filter: Callable[[Permutations], bool] | None = None) -> None:
    """
    Validate the unary operator on several `Permutations` instances.

    :param op1: the operator or operator factory
    :param number_of_samples: the optional number of samples
    :param min_unique_samples: the optional unique samples
    :param perm_filter: an optional filter to sort out permutations we cannot
        use for testing
    """
    for pwr in permutations_for_tests(perm_filter):
        validate_op1_on_1_permutations(op1, pwr, number_of_samples,
                                       min_unique_samples)


def validate_op1_with_step_size_on_1_permutations(
        op1: Op1WithStepSize | Callable[[Permutations], Op1WithStepSize],
        search_space: Permutations,
        number_of_samples: int | None = None,
        min_unique_samples: int | Callable[[
            int, Permutations], int] | None = None,
        step_sizes: Iterable[float] | Callable[
            [Permutations], Iterable[float]] = (),
        get_step_size: Callable[[
            Permutations, np.ndarray, np.ndarray,
        ], float | None] | None = None) -> None:
    """
    Validate the step-sized unary operator on one `Permutations` instance.

    :param op1: the operator or operator factory
    :param search_space: the search space
    :param number_of_samples: the optional number of samples
    :param min_unique_samples: the optional unique samples
    :param step_sizes: the step sizes to test
    :param get_step_size: try to get the step size from two space elements
    """
    args: dict[str, Any] = {
        "op1": op1(search_space) if callable(op1) else op1,
        "search_space": search_space,
        "make_search_space_element_valid":
            make_permutation_valid(search_space),
        "step_sizes": step_sizes(search_space) if callable(step_sizes)
        else step_sizes,
        "get_step_size": get_step_size,
    }
    if number_of_samples is not None:
        args["number_of_samples"] = number_of_samples
    if min_unique_samples is not None:
        args["min_unique_samples"] = min_unique_samples
    validate_op1_with_step_size(**args)


def validate_op1_with_step_size_on_permutations(
        op1: Op1WithStepSize | Callable[[Permutations], Op1WithStepSize],
        number_of_samples: int | None = None,
        min_unique_samples: int | Callable[[
            int, Permutations], int] | None = None,
        step_sizes: Iterable[float] | Callable[
            [Permutations], Iterable[float]] = (),
        get_step_size: Callable[[
            Permutations, np.ndarray, np.ndarray,
        ], float | None] | None = None,
        perm_filter: Callable[[Permutations], bool] | None = None) -> None:
    """
    Validate the unary operator on several `Permutations` instances.

    :param op1: the operator or operator factory
    :param number_of_samples: the optional number of samples
    :param min_unique_samples: the optional unique samples
    :param step_sizes: the step sizes to test
    :param get_step_size: try to get the step size from two space elements
    :param perm_filter: an optional filter to sort out permutations we cannot
        use for testing
    """
    for pwr in permutations_for_tests(perm_filter):
        validate_op1_with_step_size_on_1_permutations(
            op1, pwr, number_of_samples, min_unique_samples, step_sizes,
            get_step_size)


def validate_op2_on_1_permutations(
        op2: Op2 | Callable[[Permutations], Op2],
        search_space: Permutations,
        number_of_samples: int | None = None,
        min_unique_samples: int | Callable[[
            int, Permutations], int] | None = None) -> None:
    """
    Validate the binary operator on one `Permutations` instance.

    :param op2: the operator or operator factory
    :param search_space: the search space
    :param number_of_samples: the optional number of samples
    :param min_unique_samples: the optional unique samples
    """
    args: dict[str, Any] = {
        "op2": op2(search_space) if callable(op2) else op2,
        "search_space": search_space,
        "make_search_space_element_valid":
            make_permutation_valid(search_space),
    }
    if number_of_samples is not None:
        args["number_of_samples"] = number_of_samples
    if min_unique_samples is not None:
        args["min_unique_samples"] = min_unique_samples
    validate_op2(**args)


def validate_op2_on_permutations(
        op2: Op2 | Callable[[Permutations], Op2],
        number_of_samples: int | None = None,
        min_unique_samples: int | Callable[[
            int, Permutations], int] | None = None,
        perm_filter: Callable[[Permutations], bool] | None = None) -> None:
    """
    Validate the binary operator on several `Permutations` instances.

    :param op2: the operator or operator factory
    :param number_of_samples: the optional number of samples
    :param min_unique_samples: the optional unique samples
    :param perm_filter: an optional filter to sort out permutations we cannot
        use for testing
    """
    for pwr in permutations_for_tests(perm_filter):
        validate_op2_on_1_permutations(op2, pwr, number_of_samples,
                                       min_unique_samples)
