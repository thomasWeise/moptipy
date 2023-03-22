"""Test stuff on ordered choices-based spaces."""

from typing import Any, Callable, Iterable, cast

import numpy as np
from numpy.random import Generator, default_rng

from moptipy.api.operators import Op0
from moptipy.spaces.ordered_choices import OrderedChoices
from moptipy.tests.op0 import validate_op0
from moptipy.utils.types import type_error


def choices_for_tests(
        choice_filter: Callable[[OrderedChoices], bool] | None = None) \
        -> Iterable[OrderedChoices]:
    """
    Get a sequence of ordered choices for tests.

    :param choice_filter: an optional filter to sort out ordered choices we
        cannot use for testing
    :returns: the sequence of ordered choices
    """
    r = default_rng()
    pwrs: list[OrderedChoices] = [
        OrderedChoices([[1], [2]]),
        OrderedChoices([[1], [2], [3]]),
        OrderedChoices([[1], [2, 4], [3]]),
        OrderedChoices.signed_permutations(1),
        OrderedChoices.signed_permutations(2),
        OrderedChoices.signed_permutations(10)]

    for _i in range(4):
        done: set[int] = set()
        created: list[list[int]] = []
        choices: list[list[int]] = []
        while (len(choices) <= 0) or (len(done) < 2) or (r.integers(6) > 0):
            if r.integers(2) <= 0 < len(created):
                choices.append(created[r.integers(len(created))])
            else:
                picked: list[int] = []
                while (len(picked) <= 0) or (r.integers(6) > 0):
                    cc = r.integers(-100, 100)
                    while cc in done:
                        cc += 1
                    done.add(cc)
                    picked.append(int(cc))
                choices.append(picked)
                created.append(picked)
        pwrs.append(OrderedChoices(choices))

    if choice_filter is not None:
        if not callable(choice_filter):
            raise type_error(choice_filter, "choice_filter", None, call=True)
        pwrs = [p for p in pwrs if choice_filter(p)]
    r.shuffle(cast(list, pwrs))
    return pwrs


def make_choices_valid(choices: OrderedChoices) -> \
        Callable[[Generator, np.ndarray], np.ndarray]:
    """
    Create a function that can make ordered choices valid.

    :param choices: the ordered choices
    :returns: the function
    """

    def __make_valid(prnd: Generator, x: np.ndarray,
                     bb=choices.blueprint,
                     cc=choices.choices.__getitem__) -> np.ndarray:
        np.copyto(x, bb)
        prnd.shuffle(x)
        for i, e in enumerate(x):
            ff = cc(e)
            x[i] = ff[prnd.integers(len(ff))]
        return x

    return __make_valid


def validate_op0_on_1_choices(
        op0: Op0 | Callable[[OrderedChoices], Op0],
        search_space: OrderedChoices,
        number_of_samples: int | None = None,
        min_unique_samples: int | Callable[[
            int, OrderedChoices], int] | None = None) -> None:
    """
    Validate the nullary operator on one `OrderedChoices` instance.

    :param op0: the operator or operator factory
    :param search_space: the search space
    :param number_of_samples: the optional number of samples
    :param min_unique_samples: the optional unique samples
    """
    args: dict[str, Any] = {
        "op0": op0(search_space) if callable(op0) else op0,
        "search_space": search_space,
        "make_search_space_element_valid":
            make_choices_valid(search_space),
    }
    if number_of_samples is not None:
        args["number_of_samples"] = number_of_samples
    if min_unique_samples is not None:
        args["min_unique_samples"] = min_unique_samples
    validate_op0(**args)


def validate_op0_on_choices(
        op0: Op0 | Callable[[OrderedChoices], Op0],
        number_of_samples: int | None = None,
        min_unique_samples: int | Callable[[
            int, OrderedChoices], int] | None = None,
        choice_filter: Callable[[OrderedChoices], bool] | None = None) \
        -> None:
    """
    Validate the nullary operator on several `OrderedChoices` instances.

    :param op0: the operator or operator factory
    :param number_of_samples: the optional number of samples
    :param min_unique_samples: the optional unique samples
    :param choice_filter: an optional filter to sort out ordered choices we
        cannot use for testing
    """
    for choices in choices_for_tests(choice_filter):
        validate_op0_on_1_choices(op0, choices, number_of_samples,
                                  min_unique_samples)
