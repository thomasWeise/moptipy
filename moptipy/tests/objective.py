"""Functions that can be used to test objective functions."""
from typing import Callable, Optional

# noinspection PyProtectedMember
from moptipy.api.objective import Objective, _check_objective
from moptipy.tests.component import check_component


def check_objective(objective: Objective,
                    create_valid: Optional[Callable] = None) -> None:
    """
    Check whether an object is a moptipy objective function.

    :param objective: the objective function to test
    :param create_valid: a method that can produce one valid element,
        or `None` if the `evaluate` function should not be tested
    :raises ValueError: if `objective` is not a valid Objective
    """
    if not isinstance(objective, Objective):
        raise ValueError("Expected to receive an instance of Objective, but "
                         f"got a {type(objective)}.")
    _check_objective(objective)
    check_component(component=objective)

    lower = objective.lower_bound()
    if not (isinstance(lower, (int, float))):
        raise ValueError("lower_bound() must return an int or float, but "
                         f"returned a {type(lower)}.")

    upper = objective.upper_bound()
    if not (isinstance(upper, (int, float))):
        raise ValueError("upper_bound() must return an int or float, but "
                         f"returned a {type(upper)}.")

    if lower > upper:
        raise ValueError("Result of lower_bound() must be <= to "
                         f"upper_bound(), but got {lower} vs. {upper}.")

    if create_valid is None:
        return

    x = create_valid()
    if x is None:
        raise ValueError("create_valid() produced None.")

    res = objective.evaluate(x)
    if not (isinstance(res, (int, float))):
        raise ValueError("evaluate(x) must return an int or float, but "
                         f"returned a {type(res)}.")

    if (res < lower) or (res > upper):
        raise ValueError("evaluate(x) must return a value in"
                         "[lower_bound(), upper_bound()], but returned "
                         f"{res} vs. [{lower},{upper}].")
