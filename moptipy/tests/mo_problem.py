"""Functions for testing multi-objective optimization problems."""
from math import inf, isfinite
from typing import Callable, Optional, Any, Union, Final, Tuple, List

import numpy as np
from numpy.random import Generator, default_rng

from moptipy.api.mo_problem import MOProblem, check_mo_problem
from moptipy.api.space import Space
from moptipy.tests.objective import validate_objective
from moptipy.utils.nputils import is_np_int, is_np_float
from moptipy.utils.types import type_error


def validate_mo_problem(
        mo_problem: MOProblem,
        solution_space: Optional[Space] = None,
        make_solution_space_element_valid:
        Optional[Callable[[Generator, Any], Any]] = lambda _, x: x,
        is_deterministic: bool = True,
        lower_bound_threshold: Union[int, float] = -inf,
        upper_bound_threshold: Union[int, float] = inf,
        must_be_equal_to: Optional[
            Callable[[Any], Union[int, float]]] = None) -> None:
    """
    Check whether an object is a moptipy multi-objective optimization problem.

    :param mo_problem: the multi-objective optimization problem to test
    :param solution_space: the solution space
    :param make_solution_space_element_valid: a function that makes an element
        from the solution space valid
    :param bool is_deterministic: is the objective function deterministic?
    :param lower_bound_threshold: the threshold for the lower bound
    :param upper_bound_threshold: the threshold for the upper bound
    :param must_be_equal_to: an optional function that should return the
        exactly same values as the objective function
    :raises ValueError: if `mo_problem` is not a valid
        :class:`~moptipy.api.mo_problem.MOProblem`
    :raises TypeError: if values of the wrong types are encountered
    """
    if not isinstance(mo_problem, MOProblem):
        raise type_error(mo_problem, "mo_problem", MOProblem)
    check_mo_problem(mo_problem)
    validate_objective(mo_problem, solution_space,
                       make_solution_space_element_valid, is_deterministic,
                       lower_bound_threshold, upper_bound_threshold,
                       must_be_equal_to)

    dim: Final[int] = mo_problem.f_dimension()
    if not isinstance(dim, int):
        raise type_error(dim, "f_dimension()", int)
    if dim < 1:
        raise ValueError(f"f_dimension()=={dim} is wrong, must be >= 1")

    all_int: Final[bool] = mo_problem.is_always_integer()

    fses: Final[Tuple[np.ndarray, np.ndarray]] = \
        mo_problem.f_create(), mo_problem.f_create()

    exp_dtype: Final[np.dtype] = mo_problem.f_dtype()
    if not isinstance(exp_dtype, np.dtype):
        raise type_error(exp_dtype, "exp_dtype", np.dtype)

    if is_np_float(exp_dtype):
        if all_int:
            raise ValueError(f"if f_dtype()=={exp_dtype}, "
                             f"is_always_integer() must not be {all_int}")
    elif not is_np_int(exp_dtype):
        raise ValueError(f"f_dtype() cannot be {exp_dtype}")

    shape: Final[Tuple[int]] = (dim, )
    for fs in fses:
        if not isinstance(fs, np.ndarray):
            raise type_error(fs, "f_create()", np.ndarray)
        if len(fs) != dim:
            raise ValueError(
                f"len(f_create()) == {len(fs)} but f_dimension()=={dim}.")
        if fs.shape != shape:
            raise ValueError(
                f"f_create().shape={fs.shape}, but must be {shape}.")
        if fs.dtype != exp_dtype:
            raise ValueError(
                f"f_dtype()={exp_dtype} but f_create().dtype={fs.dtype}.")
        if not isinstance(all_int, bool):
            raise type_error(all_int, "is_always_integer()", bool)
        is_int: bool = is_np_int(fs.dtype)
        if not isinstance(is_int, bool):
            raise type_error(is_np_int, "is_np_int(dtype)", bool)
        is_float: bool = is_np_float(fs.dtype)
        if not isinstance(is_float, bool):
            raise type_error(is_float, "is_np_float(dtype)", bool)
        if not (is_int ^ is_float):
            raise ValueError(f"dtype ({fs.dtype}) of f_create() must be "
                             f"either int ({is_int}) or float ({is_float}).")
        if all_int and not is_int:
            raise ValueError(f"if is_always_integer()=={all_int}, then the "
                             f"dtype ({fs.dtype}) of f_create() must be an "
                             f"integer type, but is not ({is_int}).")
    fs1: np.ndarray
    fs2: np.ndarray
    fs1, fs2 = fses
    if fs1.dtype is not fs2.dtype:
        raise ValueError("encountered two different dtypes when invoking "
                         f"f_create() twice: {fs1.dtype}, {fs2.dtype}")

    lower: Final[Union[int, float]] = mo_problem.lower_bound()
    if not (isinstance(lower, (int, float))):
        raise type_error(lower, "lower_bound()", (int, float))
    if (not isfinite(lower)) and (not (lower <= (-inf))):
        raise ValueError(
            f"lower bound must be finite or -inf, but is {lower}.")
    if lower < lower_bound_threshold:
        raise ValueError("lower bound must not be less than "
                         f"{lower_bound_threshold}, but is {lower}.")

    upper: Final[Union[int, float]] = mo_problem.upper_bound()
    if not (isinstance(upper, (int, float))):
        raise type_error(upper, "upper_bound()", (int, float))
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
    random: Final[Generator] = default_rng()
    x = make_solution_space_element_valid(random, x)
    if x is None:
        raise ValueError("make_solution_space_element_valid() produced None.")
    solution_space.validate(x)

    reses: Final[List[Union[int, float]]] = [
        mo_problem.f_evaluate(x, fs1), mo_problem.f_evaluate(x, fs2)]
    if len(reses) != 2:
        raise ValueError(f"Huh? {len(reses)} != 2 for {reses}??")

    for fs in fses:
        for v in fs:
            if not isfinite(v):
                raise ValueError(f"encountered non-finite value {v} in "
                                 f"objective vector {fs} of {x}.")
        mo_problem.f_validate(fs)

    fdr = mo_problem.f_dominates(fses[0], fses[1])
    if fdr != 2:
        raise ValueError(f"f_dominates(x, x) must be 2, but is {fdr}")

    for res in reses:
        if not isinstance(res, (int, float)):
            raise type_error(res, "f_evaluate(x)", (int, float))
        if not isfinite(res):
            raise ValueError(
                f"result of f_evaluate() must be finite, but is {res}.")
        if res < lower:
            raise ValueError(f"f_evaluate()={res} < lower_bound()={lower}")
        if res > upper:
            raise ValueError(f"f_evaluate()={res} > upper_bound()={upper}")
    if is_deterministic:
        if not np.array_equal(fs1, fs2):
            raise ValueError("deterministic objective returns vectors "
                             f"{fses} when evaluating {x}.")
        if reses[0] != reses[1]:
            raise ValueError("deterministic objective returns scalar "
                             f"{reses} when evaluating {x}.")
