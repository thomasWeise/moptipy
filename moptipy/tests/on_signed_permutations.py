"""Test stuff on signed permutations with repetitions."""

from typing import Any, Callable, Iterable, cast

import numpy as np
from numpy.random import Generator, default_rng

from moptipy.api.operators import Op0, Op1
from moptipy.spaces.signed_permutations import SignedPermutations
from moptipy.tests.op0 import validate_op0
from moptipy.tests.op1 import validate_op1
from moptipy.utils.types import type_error


def signed_permutations_for_tests(
        perm_filter: Callable[[SignedPermutations], bool] | None = None) \
        -> Iterable[SignedPermutations]:
    """
    Get a sequence of permutations for tests.

    :param perm_filter: an optional filter to sort out permutations we cannot
        use for testing
    :returns: the sequence of SignedPermutations
    """
    r = default_rng()
    pwrs: list[SignedPermutations] = [
        SignedPermutations.standard(2),
        SignedPermutations.standard(3),
        SignedPermutations.standard(4),
        SignedPermutations.standard(5),
        SignedPermutations.standard(6),
        SignedPermutations.standard(12),
        SignedPermutations.standard(23),
        SignedPermutations.with_repetitions(2, 2),
        SignedPermutations.with_repetitions(2, 3),
        SignedPermutations.with_repetitions(3, 2),
        SignedPermutations.with_repetitions(3, 3),
        SignedPermutations.with_repetitions(5, 5),
        SignedPermutations.with_repetitions(int(r.integers(6, 10)),
                                            int(r.integers(2, 7))),
        SignedPermutations.with_repetitions(int(r.integers(2, 5)),
                                            int(r.integers(6, 10))),
        SignedPermutations.with_repetitions(int(r.integers(130, 500)),
                                            int(r.integers(2, 200))),
        SignedPermutations([1, 1, 1, 1, 1, 5, 5, 3]),
        SignedPermutations([2, 1, 1, 1, 1, 1])]
    if perm_filter is not None:
        if not callable(perm_filter):
            raise type_error(perm_filter, "perm_filter", None, call=True)
        pwrs = [p for p in pwrs if perm_filter(p)]
    r.shuffle(cast(list, pwrs))
    return pwrs


def make_signed_permutation_valid(pwr: SignedPermutations) -> \
        Callable[[Generator, np.ndarray], np.ndarray]:
    """
    Create a function that can make permutations with repetitions valid.

    :param pwr: the permutations
    :returns: the function
    """
    def __make_valid(prnd: Generator, x: np.ndarray, ppp=pwr) -> np.ndarray:
        np.copyto(x, ppp.blueprint)
        prnd.shuffle(x)
        x *= ((2 * prnd.integers(low=0, high=2, size=len(x))) - 1)
        return x
    return __make_valid


def validate_op0_on_1_signed_permutations(
        op0: Op0 | Callable[[SignedPermutations], Op0],
        search_space: SignedPermutations,
        number_of_samples: int | None = None,
        min_unique_samples: int | Callable[[
            int, SignedPermutations], int] | None = None) -> None:
    """
    Validate the nullary operator on one `SignedPermutations` instance.

    :param op0: the operator or operator factory
    :param search_space: the search space
    :param number_of_samples: the optional number of samples
    :param min_unique_samples: the optional unique samples
    """
    args: dict[str, Any] = {
        "op0": op0(search_space) if callable(op0) else op0,
        "search_space": search_space,
        "make_search_space_element_valid":
            make_signed_permutation_valid(search_space),
    }
    if number_of_samples is not None:
        args["number_of_samples"] = number_of_samples
    if min_unique_samples is not None:
        args["min_unique_samples"] = min_unique_samples
    validate_op0(**args)


def validate_op0_on_signed_permutations(
        op0: Op0 | Callable[[SignedPermutations], Op0],
        number_of_samples: int | None = None,
        min_unique_samples: int | Callable[[
            int, SignedPermutations], int] | None = None,
        perm_filter: Callable[[SignedPermutations], bool] | None = None) \
        -> None:
    """
    Validate the nullary operator on several `SignedPermutations` instances.

    :param op0: the operator or operator factory
    :param number_of_samples: the optional number of samples
    :param min_unique_samples: the optional unique samples
    :param perm_filter: an optional filter to sort out permutations we cannot
        use for testing
    """
    for pwr in signed_permutations_for_tests(perm_filter):
        validate_op0_on_1_signed_permutations(op0, pwr, number_of_samples,
                                              min_unique_samples)


def validate_op1_on_1_signed_permutations(
        op1: Op1 | Callable[[SignedPermutations], Op1],
        search_space: SignedPermutations,
        number_of_samples: int | None = None,
        min_unique_samples: int | Callable[[
            int, SignedPermutations], int] | None = None) -> None:
    """
    Validate the unary operator on one `SignedPermutations` instance.

    :param op1: the operator or operator factory
    :param search_space: the search space
    :param number_of_samples: the optional number of samples
    :param min_unique_samples: the optional unique samples
    """
    args: dict[str, Any] = {
        "op1": op1(search_space) if callable(op1) else op1,
        "search_space": search_space,
        "make_search_space_element_valid":
            make_signed_permutation_valid(search_space),
    }
    if number_of_samples is not None:
        args["number_of_samples"] = number_of_samples
    if min_unique_samples is not None:
        args["min_unique_samples"] = min_unique_samples
    validate_op1(**args)


def validate_op1_on_signed_permutations(
        op1: Op1 | Callable[[SignedPermutations], Op1],
        number_of_samples: int | None = None,
        min_unique_samples: int | Callable[[
            int, SignedPermutations], int] | None = None,
        perm_filter: Callable[[SignedPermutations], bool] | None = None) \
        -> None:
    """
    Validate the unary operator on several `SignedPermutations` instances.

    :param op1: the operator or operator factory
    :param number_of_samples: the optional number of samples
    :param min_unique_samples: the optional unique samples
    :param perm_filter: an optional filter to sort out permutations we cannot
        use for testing
    """
    for pwr in signed_permutations_for_tests(perm_filter):
        validate_op1_on_1_signed_permutations(op1, pwr, number_of_samples,
                                              min_unique_samples)
