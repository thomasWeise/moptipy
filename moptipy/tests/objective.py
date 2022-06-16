"""Functions that can be used to test objective functions."""
from math import isfinite, inf
from typing import Callable, Optional, Any, Union

from numpy.random import default_rng, Generator

from moptipy.api.objective import Objective, check_objective
from moptipy.api.space import Space
from moptipy.tests.component import validate_component


def validate_objective(
        objective: Objective,
        solution_space: Optional[Space] = None,
        make_solution_space_element_valid:
        Optional[Callable[[Generator, Any], Any]] = lambda _, x: x,
        is_deterministic: bool = True,
        lower_bound_threshold: Union[int, float] = -inf,
        upper_bound_threshold: Union[int, float] = inf,
        must_be_equal_to: Optional[
            Callable[[Any], Union[int, float]]] = None) -> None:
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
    :raises ValueError: if `objective` is not a valid Objective
    """
    if not isinstance(objective, Objective):
        raise ValueError("Expected to receive an instance of Objective, but "
                         f"got a {type(objective)}.")
    check_objective(objective)
    validate_component(objective)

    if not (hasattr(objective, 'lower_bound')
            and callable(getattr(objective, 'lower_bound'))):
        raise ValueError("objective must have method lower_bound.")
    lower = objective.lower_bound()
    if not (isinstance(lower, (int, float))):
        raise ValueError("lower_bound() must return an int or float, but "
                         f"returned a {type(lower)}.")
    if (not isfinite(lower)) and (not (lower <= (-inf))):
        raise ValueError(
            f"lower bound must be finite or -inf, but is {lower}.")
    if lower < lower_bound_threshold:
        raise ValueError("lower bound must not be less than "
                         f"{lower_bound_threshold}, but is {lower}.")

    if not (hasattr(objective, 'upper_bound')
            and callable(getattr(objective, 'upper_bound'))):
        raise ValueError("objective must have method upper_bound.")
    upper = objective.upper_bound()
    if not (isinstance(upper, (int, float))):
        raise ValueError("upper_bound() must return an int or float, but "
                         f"returned a {type(upper)}.")
    if (not isfinite(upper)) and (not (upper >= inf)):
        raise ValueError(
            f"upper bound must be finite or +inf, but is {upper}.")
    if upper > upper_bound_threshold:
        raise ValueError(
            f"upper bound must not be more than {upper_bound_threshold}, "
            f"but is {lower}.")

    if lower >= upper:
        raise ValueError("Result of lower_bound() must be smaller than "
                         f"upper_bound(), but got {lower} vs. {upper}.")

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

    if not (hasattr(objective, 'evaluate')
            and callable(getattr(objective, 'evaluate'))):
        raise ValueError("objective must have method evaluate.")
    res = objective.evaluate(x)
    if not (isinstance(res, (int, float))):
        raise ValueError(f"evaluate(x) of {x} must return an int or float, "
                         f"but returned a {type(res)}.")

    if (res < lower) or (res > upper):
        raise ValueError(f"evaluate(x) of {x} must return a value in"
                         "[lower_bound(), upper_bound()], but returned "
                         f"{res} vs. [{lower},{upper}].")
    if must_be_equal_to is not None:
        exp = must_be_equal_to(x)
        if exp != res:
            raise ValueError(f"expected to get {exp}, but got {res}.")

    res2 = objective.evaluate(x)
    if not (isinstance(res2, (int, float))):
        raise ValueError(f"evaluate(x) of {x} must return an int or float, "
                         f"but returned a {type(res2)}.")

    if (res2 < lower) or (res2 > upper):
        raise ValueError(f"evaluate(x) of {x} must return a value in"
                         "[lower_bound(), upper_bound()], but returned "
                         f"{res2} vs. [{lower},{upper}].")

    if is_deterministic and (res != res2):
        raise ValueError(f"evaluating {x} twice yielded the two different "
                         f"results {res} and {res2}!")
