"""Functions that can be used to test objective functions."""
from typing import Callable, Optional
from math import isfinite, inf

import moptipy.api.objective as mo
from moptipy.tests.component import test_component


def test_objective(objective: mo.Objective,
                   create_valid: Optional[Callable] = None) -> None:
    """
    Check whether an object is a moptipy objective function.

    :param objective: the objective function to test
    :param create_valid: a method that can produce one valid element,
        or `None` if the `evaluate` function should not be tested
    :raises ValueError: if `objective` is not a valid Objective
    """
    if not isinstance(objective, mo.Objective):
        raise ValueError("Expected to receive an instance of Objective, but "
                         f"got a {type(objective)}.")
    mo.check_objective(objective)
    test_component(component=objective)

    lower = objective.lower_bound()
    if not (isinstance(lower, (int, float))):
        raise ValueError("lower_bound() must return an int or float, but "
                         f"returned a {type(lower)}.")
    if (not isfinite(lower)) and (not (lower <= (-inf))):
        raise ValueError(
            f"lower bound must be finite or -inf, but is {lower}.")

    upper = objective.upper_bound()
    if not (isinstance(upper, (int, float))):
        raise ValueError("upper_bound() must return an int or float, but "
                         f"returned a {type(upper)}.")
    if (not isfinite(upper)) and (not (upper >= inf)):
        raise ValueError(
            f"upper bound must be finite or +inf, but is {upper}.")

    if lower >= upper:
        raise ValueError("Result of lower_bound() must be smaller than "
                         f"upper_bound(), but got {lower} vs. {upper}.")

    if create_valid is None:
        return

    x = create_valid()
    if x is None:
        raise ValueError("create_valid() produced None.")

    res = objective.evaluate(x)
    if not (isinstance(res, (int, float))):
        raise ValueError(f"evaluate(x) of {x} must return an int or float, "
                         f"but returned a {type(res)}.")

    if (res < lower) or (res > upper):
        raise ValueError(f"evaluate(x) of {x} must return a value in"
                         "[lower_bound(), upper_bound()], but returned "
                         f"{res} vs. [{lower},{upper}].")

    res2 = objective.evaluate(x)
    if res != res2:
        raise ValueError(f"evaluating {x} twice yielded the two different "
                         f"results {res} and {res2}!")
