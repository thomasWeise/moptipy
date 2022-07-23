"""Test stuff on bit strings."""

from typing import Callable, Union, Iterable, List, Optional, Dict, Any, \
    cast, Final

import numpy as np
from numpy.random import default_rng, Generator

from moptipy.api.algorithm import Algorithm
from moptipy.api.objective import Objective
from moptipy.api.operators import Op0, Op1, Op2
from moptipy.examples.bitstrings.leadingones import LeadingOnes
from moptipy.examples.bitstrings.onemax import OneMax
from moptipy.spaces.bitstrings import BitStrings
from moptipy.tests.algorithm import validate_algorithm
from moptipy.tests.op0 import validate_op0
from moptipy.tests.op1 import validate_op1
from moptipy.tests.op2 import validate_op2
from moptipy.utils.types import type_error


def dimensions_for_tests() -> Iterable[int]:
    """
    Get a sequence of dimensions for tests.

    :returns: the sequence of integers
    """
    r = default_rng()
    bs: List[int] = [1, 2, 3, 4, 5, 10, 16, 100,
                     int(r.integers(20, 50)), int(r.integers(200, 300))]
    r.shuffle(cast(List, bs))
    return bs


def bitstrings_for_tests() -> Iterable[BitStrings]:
    """
    Get a sequence of bit strings for tests.

    :returns: the sequence of BitStrings
    """
    return [BitStrings(i) for i in dimensions_for_tests()]


def random_bit_string(random: Generator, x: np.ndarray) -> np.ndarray:
    """
    Randomize a bit string.

    :param random: the random number generator
    :param x: the bit string
    :returns: the array
    """
    ri = random.integers
    for i in range(len(x)):  # pylint: disable=C0200
        x[i] = ri(2) <= 0
    return x


def validate_op0_on_1_bitstrings(
        op0: Union[Op0, Callable[[BitStrings], Op0]],
        search_space: BitStrings,
        number_of_samples: Optional[int] = None,
        min_unique_samples:
        Optional[Union[int, Callable[[int, BitStrings], int]]]
        = None) -> None:
    """
    Validate the unary operator on one bit strings instance.

    :param op0: the operator or operator factory
    :param search_space: the search space
    :param number_of_samples: the optional number of samples
    :param min_unique_samples: the optional unique samples
    """
    args: Dict[str, Any] = {
        "op0": op0(search_space) if callable(op0) else op0,
        "search_space": search_space,
        "make_search_space_element_valid": random_bit_string
    }
    if number_of_samples is not None:
        args["number_of_samples"] = number_of_samples
    if min_unique_samples is not None:
        args["min_unique_samples"] = min_unique_samples
    validate_op0(**args)


def validate_op0_on_bitstrings(
        op0: Union[Op0, Callable[[BitStrings], Op0]],
        number_of_samples: Optional[int] = None,
        min_unique_samples:
        Optional[Union[int, Callable[[int, BitStrings], int]]]
        = None) -> None:
    """
    Validate the unary operator on several BitStrings instances.

    :param op0: the operator or operator factory
    :param number_of_samples: the optional number of samples
    :param min_unique_samples: the optional unique samples
    """
    for bst in bitstrings_for_tests():
        validate_op0_on_1_bitstrings(op0, bst,
                                     number_of_samples, min_unique_samples)


def validate_op1_on_1_bitstrings(
        op1: Union[Op1, Callable[[BitStrings], Op1]],
        search_space: BitStrings,
        number_of_samples: Optional[int] = None,
        min_unique_samples:
        Optional[Union[int, Callable[[int, BitStrings], int]]]
        = None) -> None:
    """
    Validate the unary operator on one BitStrings instance.

    :param op1: the operator or operator factory
    :param search_space: the search space
    :param number_of_samples: the optional number of samples
    :param min_unique_samples: the optional unique samples
    """
    args: Dict[str, Any] = {
        "op1": op1(search_space) if callable(op1) else op1,
        "search_space": search_space,
        "make_search_space_element_valid": random_bit_string
    }
    if number_of_samples is not None:
        args["number_of_samples"] = number_of_samples
    if min_unique_samples is not None:
        args["min_unique_samples"] = min_unique_samples
    validate_op1(**args)


def validate_op1_on_bitstrings(
        op1: Union[Op1, Callable[[BitStrings], Op1]],
        number_of_samples: Optional[int] = None,
        min_unique_samples:
        Optional[Union[int, Callable[[int, BitStrings], int]]]
        = None) -> None:
    """
    Validate the unary operator on several BitStrings instances.

    :param op1: the operator or operator factory
    :param number_of_samples: the optional number of samples
    :param min_unique_samples: the optional unique samples
    """
    for bst in bitstrings_for_tests():
        validate_op1_on_1_bitstrings(op1, bst,
                                     number_of_samples, min_unique_samples)


def validate_op2_on_1_bitstrings(
        op2: Union[Op2, Callable[[BitStrings], Op2]],
        search_space: BitStrings,
        number_of_samples: Optional[int] = None,
        min_unique_samples:
        Optional[Union[int, Callable[[int, BitStrings], int]]]
        = None) -> None:
    """
    Validate the binary operator on one BitStrings instance.

    :param op2: the operator or operator factory
    :param search_space: the search space
    :param number_of_samples: the optional number of samples
    :param min_unique_samples: the optional unique samples
    """
    args: Dict[str, Any] = {
        "op2": op2(search_space) if callable(op2) else op2,
        "search_space": search_space,
        "make_search_space_element_valid": random_bit_string
    }
    if number_of_samples is not None:
        args["number_of_samples"] = number_of_samples
    if min_unique_samples is not None:
        args["min_unique_samples"] = min_unique_samples
    validate_op2(**args)


def validate_op2_on_bitstrings(
        op2: Union[Op2, Callable[[BitStrings], Op2]],
        number_of_samples: Optional[int] = None,
        min_unique_samples:
        Optional[Union[int, Callable[[int, BitStrings], int]]]
        = None) -> None:
    """
    Validate the binary operator on several BitStrings instances.

    :param op2: the operator or operator factory
    :param number_of_samples: the optional number of samples
    :param min_unique_samples: the optional unique samples
    """
    for bst in bitstrings_for_tests():
        validate_op2_on_1_bitstrings(op2, bst,
                                     number_of_samples, min_unique_samples)


def validate_algorithm_on_bitstrings(
        objective: Union[Objective, Callable[[int], Objective]],
        algorithm: Union[Algorithm,
                         Callable[[BitStrings], Algorithm]],
        dimension: int = 5,
        max_fes: int = 100,
        required_result: Optional[Union[int, Callable[
            [int, int], int]]] = None) -> None:
    """
    Check the validity of a black-box algorithm on a bit strings problem.

    :param algorithm: the algorithm or algorithm factory
    :param objective: the instance or instance factory
    :param dimension: the dimension of the problem
    :param max_fes: the maximum number of FEs
    :param required_result: the optional required result quality
    """
    if not (isinstance(algorithm, Algorithm) or callable(algorithm)):
        raise type_error(algorithm, 'algorithm', Algorithm, True)
    if not (isinstance(objective, Objective) or callable(objective)):
        raise type_error(objective, "objective", Objective, True)
    if not isinstance(dimension, int):
        raise type_error(dimension, 'dimension', int)
    if dimension <= 0:
        raise ValueError(f"dimension must be > 0, but got {dimension}.")

    if callable(objective):
        objective = objective(dimension)
    if not isinstance(objective, Objective):
        raise type_error(objective, "result of callable 'objective'",
                         Objective)
    bs: Final[BitStrings] = BitStrings(dimension)
    if callable(algorithm):
        algorithm = algorithm(bs)
    if not isinstance(algorithm, Algorithm):
        raise type_error(algorithm, "result of callable 'algorithm'",
                         Algorithm)

    goal: Optional[int]
    if callable(required_result):
        goal = required_result(max_fes, dimension)
    else:
        goal = required_result

    validate_algorithm(algorithm=algorithm,
                       solution_space=bs,
                       objective=objective,
                       max_fes=max_fes,
                       required_result=goal)


def validate_algorithm_on_onemax(
        algorithm: Union[Algorithm,
                         Callable[[BitStrings], Algorithm]]) -> None:
    """
    Check the validity of a black-box algorithm on onemax.

    :param algorithm: the algorithm or algorithm factory
    """
    max_fes: Final[int] = 100
    for i in dimensions_for_tests():
        rr: int
        if i < 3:
            rr = 1
        else:
            rr = max(1, i // 2, i - int(max_fes ** 0.5))
        validate_algorithm_on_bitstrings(
            objective=OneMax,
            algorithm=algorithm,
            dimension=i,
            max_fes=max_fes,
            required_result=rr)


def validate_algorithm_on_leadingones(
        algorithm: Union[Algorithm,
                         Callable[[BitStrings], Algorithm]]) -> None:
    """
    Check the validity of a black-box algorithm on leadingones.

    :param algorithm: the algorithm or algorithm factory
    """
    max_fes: Final[int] = 100
    for i in dimensions_for_tests():
        rr: int
        if i < 3:
            rr = 0
        elif max_fes > (10 * (i ** 1.5)):
            rr = i - 1
        else:
            rr = i
        validate_algorithm_on_bitstrings(
            objective=LeadingOnes,
            algorithm=algorithm,
            dimension=i,
            max_fes=int(1.25 * max_fes),
            required_result=rr)
