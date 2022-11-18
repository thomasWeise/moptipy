"""Functions that can be used to test objective functions."""
from math import inf, isfinite
from typing import Any, Callable, Final

from numpy.random import Generator, default_rng

from moptipy.api.objective import Objective, check_objective
from moptipy.api.space import Space
from moptipy.tests.component import validate_component
from moptipy.utils.types import type_error


def validate_objective(
        objective: Objective,
        solution_space: Space | None = None,
        make_solution_space_element_valid:
        Callable[[Generator, Any], Any] | None = lambda _, x: x,
        is_deterministic: bool = True,
        lower_bound_threshold: int | float = -inf,
        upper_bound_threshold: int | float = inf,
        must_be_equal_to: Callable[[Any], int | float] | None = None) -> None:
    """
    Check whether an object is a moptipy objective function.

    :param objective: the objective function to test
    :param solution_space: the solution space
    :param make_solution_space_element_valid: a function that makes an element
        from the solution space valid
    :param bool is_deterministic: is the objective function deterministic?
    :param lower_bound_threshold: the threshold for the lower bound
    :param upper_bound_threshold: the threshold for the upper bound
    :param must_be_equal_to: an optional function that should return the
        exactly same values as the objective function
    :raises ValueError: if `objective` is not a valid
        :class:`~moptipy.api.objective.Objective`
    :raises TypeError: if values of the wrong types are encountered
    """
    if not isinstance(objective, Objective):
        raise type_error(objective, "objective", Objective)
    check_objective(objective)
    validate_component(objective)

    if not (hasattr(objective, "lower_bound")
            and callable(getattr(objective, "lower_bound"))):
        raise ValueError("objective must have method lower_bound.")
    lower: Final[int | float] = objective.lower_bound()
    if not (isinstance(lower, (int, float))):
        raise type_error(lower, "lower_bound()", (int, float))
    if (not isfinite(lower)) and (not (lower <= (-inf))):
        raise ValueError(
            f"lower bound must be finite or -inf, but is {lower}.")
    if lower < lower_bound_threshold:
        raise ValueError("lower bound must not be less than "
                         f"{lower_bound_threshold}, but is {lower}.")

    if not (hasattr(objective, "upper_bound")
            and callable(getattr(objective, "upper_bound"))):
        raise ValueError("objective must have method upper_bound.")
    upper: Final[int | float] = objective.upper_bound()
    if not (isinstance(upper, (int, float))):
        raise type_error(upper, "upper_bound()", (int, float))
    if (not isfinite(upper)) and (not (upper >= inf)):
        raise ValueError(
            f"upper bound must be finite or +inf, but is {upper}.")
    if upper > upper_bound_threshold:
        raise ValueError(
            f"upper bound must not be more than {upper_bound_threshold}, "
            f"but is {upper}.")

    if lower >= upper:
        raise ValueError("Result of lower_bound() must be smaller than "
                         f"upper_bound(), but got {lower} vs. {upper}.")

    if not (hasattr(objective, "is_always_integer")
            and callable(getattr(objective, "is_always_integer"))):
        raise ValueError("objective must have method is_always_integer.")
    is_int: Final[bool] = objective.is_always_integer()
    if not isinstance(is_int, bool):
        raise type_error(is_int, "is_always_integer()", bool)
    if is_int:
        if isfinite(lower) and (not isinstance(lower, int)):
            raise TypeError(
                f"if is_always_integer()==True, then lower_bound() must "
                f"return int, but it returned {lower}.")
        if isfinite(upper) and (not isinstance(upper, int)):
            raise TypeError(
                f"if is_always_integer()==True, then upper_bound() must "
                f"return int, but it returned {upper}.")

    count: int = 0
    if make_solution_space_element_valid is not None:
        count += 1
    if solution_space is not None:
        count += 1
    if count <= 0:
        return
    if count < 2:
        raise ValueError("either provide both of solution_space and "
                         "make_solution_space_element_valid or none.")

    x = solution_space.create()
    if x is None:
        raise ValueError("solution_space.create() produced None.")
    x = make_solution_space_element_valid(default_rng(), x)
    if x is None:
        raise ValueError("make_solution_space_element_valid() produced None.")
    solution_space.validate(x)

    if not (hasattr(objective, "evaluate")
            and callable(getattr(objective, "evaluate"))):
        raise ValueError("objective must have method evaluate.")
    res = objective.evaluate(x)
    if not (isinstance(res, (int, float))):
        raise type_error(res, f"evaluate(x) of {x}", (int, float))

    if (res < lower) or (res > upper):
        raise ValueError(f"evaluate(x) of {x} must return a value in"
                         "[lower_bound(), upper_bound()], but returned "
                         f"{res} vs. [{lower},{upper}].")
    if is_int and (not isinstance(res, int)):
        raise TypeError(
            f"if is_always_integer()==True, then evaluate(x) must "
            f"return int, but it returned {res}.")

    if must_be_equal_to is not None:
        exp = must_be_equal_to(x)
        if exp != res:
            raise ValueError(f"expected to get {exp}, but got {res}.")

    res2 = objective.evaluate(x)
    if not (isinstance(res2, (int, float))):
        raise type_error(res2, f"evaluate(x) of {x}", (int, float))

    if (res2 < lower) or (res2 > upper):
        raise ValueError(f"evaluate(x) of {x} must return a value in"
                         "[lower_bound(), upper_bound()], but returned "
                         f"{res2} vs. [{lower},{upper}].")
    if is_int and (not isinstance(res2, int)):
        raise TypeError(
            f"if is_always_integer()==True, then evaluate(x) must "
            f"return int, but it returned {res2}.")

    if is_deterministic and (res != res2):
        raise ValueError(f"evaluating {x} twice yielded the two different "
                         f"results {res} and {res2}!")
