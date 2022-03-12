"""Test stuff on permutations with repetitions."""

from typing import Callable, Union, Iterable, List, Optional, Dict, Any, cast

import numpy as np
from numpy.random import default_rng

from moptipy.api.operators import Op0, Op1
from moptipy.spaces.permutationswr import PermutationsWithRepetitions
from moptipy.tests.op0 import validate_op0
from moptipy.tests.op1 import validate_op1


def pwrs_for_tests() -> Iterable[PermutationsWithRepetitions]:
    """
    Get a sequence of permutations with repetitions for tests.

    :returns: the sequence of PermutationsWithRepetitions
    """
    r = default_rng()
    pwrs: List[PermutationsWithRepetitions] = [
        PermutationsWithRepetitions(2, 2),
        PermutationsWithRepetitions(2, 3),
        PermutationsWithRepetitions(3, 2),
        PermutationsWithRepetitions(3, 3),
        PermutationsWithRepetitions(5, 5),
        PermutationsWithRepetitions(int(r.integers(6, 10)),
                                    int(r.integers(2, 7))),
        PermutationsWithRepetitions(int(r.integers(2, 5)),
                                    int(r.integers(6, 10))),
        PermutationsWithRepetitions(int(r.integers(130, 500)),
                                    int(r.integers(2, 200)))]
    r.shuffle(cast(List, pwrs))
    return pwrs


def make_pwr_valid(pwr: PermutationsWithRepetitions) \
        -> Callable[[np.ndarray], np.ndarray]:
    """
    Create a function that can make permutations with repetitions valid.

    :param pwr: the permutation with repetition
    :returns: the function
    """
    rnd = default_rng()

    def __make_valid(x: np.ndarray,
                     prnd=rnd,
                     n=pwr.n,
                     r=pwr.repetitions) -> np.ndarray:
        i: int = 0
        for _ in range(r):
            for k in range(n):
                x[i] = k
                i += 1
        prnd.shuffle(x)
        return x

    return __make_valid


def validate_op0_on_1_pwr(
        op0: Union[Op0, Callable[[PermutationsWithRepetitions], Op0]],
        search_space: PermutationsWithRepetitions,
        number_of_samples: Optional[int] = None,
        min_unique_samples:
        Optional[Union[int, Callable[[int,
                                      PermutationsWithRepetitions], int]]]
        = None) -> None:
    """
    Validate the unary operator on one PWR instance.

    :param op0: the operator or operator factory
    :param search_space: the search space
    :param number_of_samples: the optional number of samples
    :param min_unique_samples: the optional unique samples
    """
    args: Dict[str, Any] = {
        "op0": op0(search_space) if callable(op0) else op0,
        "search_space": search_space,
        "make_search_space_element_valid": make_pwr_valid(search_space)
    }
    if number_of_samples is not None:
        args["number_of_samples"] = number_of_samples
    if min_unique_samples is not None:
        args["min_unique_samples"] = min_unique_samples
    validate_op0(**args)


def validate_op0_on_pwr(
        op0: Union[Op0, Callable[[PermutationsWithRepetitions], Op0]],
        number_of_samples: Optional[int] = None,
        min_unique_samples:
        Optional[Union[int, Callable[[int,
                                      PermutationsWithRepetitions], int]]]
        = None) -> None:
    """
    Validate the unary operator on several PWR instances.

    :param op0: the operator or operator factory
    :param number_of_samples: the optional number of samples
    :param min_unique_samples: the optional unique samples
    """
    for pwr in pwrs_for_tests():
        validate_op0_on_1_pwr(op0, pwr, number_of_samples, min_unique_samples)


def validate_op1_on_1_pwr(
        op1: Union[Op1, Callable[[PermutationsWithRepetitions], Op1]],
        search_space: PermutationsWithRepetitions,
        number_of_samples: Optional[int] = None,
        min_unique_samples:
        Optional[Union[int, Callable[[int,
                                      PermutationsWithRepetitions], int]]]
        = None) -> None:
    """
    Validate the unary operator on one PWR instance.

    :param op1: the operator or operator factory
    :param search_space: the search space
    :param number_of_samples: the optional number of samples
    :param min_unique_samples: the optional unique samples
    """
    args: Dict[str, Any] = {
        "op1": op1(search_space) if callable(op1) else op1,
        "search_space": search_space,
        "make_search_space_element_valid": make_pwr_valid(search_space)
    }
    if number_of_samples is not None:
        args["number_of_samples"] = number_of_samples
    if min_unique_samples is not None:
        args["min_unique_samples"] = min_unique_samples
    validate_op1(**args)


def validate_op1_on_pwr(
        op1: Union[Op1, Callable[[PermutationsWithRepetitions], Op1]],
        number_of_samples: Optional[int] = None,
        min_unique_samples:
        Optional[Union[int, Callable[[int,
                                      PermutationsWithRepetitions], int]]]
        = None) -> None:
    """
    Validate the unary operator on several PWR instances.

    :param op1: the operator or operator factory
    :param number_of_samples: the optional number of samples
    :param min_unique_samples: the optional unique samples
    """
    for pwr in pwrs_for_tests():
        validate_op1_on_1_pwr(op1, pwr, number_of_samples, min_unique_samples)
