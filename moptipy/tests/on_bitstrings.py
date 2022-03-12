"""Test stuff on bit strings."""

from typing import Callable, Union, Iterable, List, Optional, Dict, Any, cast

from numpy.random import default_rng

from moptipy.api.operators import Op0, Op1
from moptipy.spaces.bitstrings import BitStrings
from moptipy.tests.op0 import validate_op0
from moptipy.tests.op1 import validate_op1


def bitstrings_for_tests() -> Iterable[BitStrings]:
    """
    Get a sequence of bit strings for tests.

    :returns: the sequence of BitStrings
    """
    r = default_rng()
    bs: List[BitStrings] = [
        BitStrings(1), BitStrings(2), BitStrings(3), BitStrings(4),
        BitStrings(5), BitStrings(10), BitStrings(16), BitStrings(100),
        BitStrings(int(r.integers(20, 50))),
        BitStrings(int(r.integers(200, 300)))]
    r.shuffle(cast(List, bs))
    return bs


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
        "search_space": search_space
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
    for pwr in bitstrings_for_tests():
        validate_op0_on_1_bitstrings(op0, pwr,
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
        "search_space": search_space
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
    for pwr in bitstrings_for_tests():
        validate_op1_on_1_bitstrings(op1, pwr,
                                     number_of_samples, min_unique_samples)
