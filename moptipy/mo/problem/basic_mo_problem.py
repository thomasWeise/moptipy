"""The base class for implementing multi-objective problems."""
from math import inf, isfinite
from typing import Any, Callable, Final, Iterable

import numpy as np
from numpy import empty

from moptipy.api.logging import KEY_SPACE_NUM_VARS, SCOPE_OBJECTIVE_FUNCTION
from moptipy.api.mo_problem import MOProblem
from moptipy.api.mo_utils import dominates
from moptipy.api.objective import Objective
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.math import try_int
from moptipy.utils.nputils import (
    KEY_NUMPY_TYPE,
    dtype_for_data,
    numpy_type_to_str,
)
from moptipy.utils.types import type_error


class BasicMOProblem(MOProblem):
    """
    The base class for implementing multi-objective optimization problems.

    This class allows to construct a simple python function for scalarizing
    a vector of objective values in its constructor and also determines the
    right datatype for the objective vectors.

    It therefore first obtains the type (integers or floats?) of the objective
    values as well as the bounds of the objective functions. This is used to
    determine the right numpy `dtype` for the objective vectors. We want to
    represent objective vectors as compact as possible and use an integer
    vector if possible.

    Once this information is obtained, we invoke a call-back function
    `get_scalarizer` which should return a python function that computes the
    scalarization result, i.e., the single scalar value representing the
    vector of objective values in single-objective optimization. This function
    must be monotonous. If the bounds are finite, it is applied to the vector
    of lower and upper bounds to get the lower and upper bounds of the
    scalarization result.

    Examples for implementing this class are
    class:`~moptipy.mo.problem.weighted_sum.WeightedSum` and
    :class:`~moptipy.mo.problem.weighted_sum.Prioritize`, which represent a
    multi-objective optimization problem either as weighted sum or by
    priorizing the objective value (via an internal weighted sum).
    """

    def __init__(self, objectives: Iterable[Objective],
                 get_scalarizer: Callable[[bool, int, list[int | float],
                                           list[int | float]],
                 Callable[[np.ndarray], int | float]] | None = None,
                 domination: Callable[[np.ndarray, np.ndarray], int] | None
                 = dominates) -> None:
        """
        Create the basic multi-objective optimization problem.

        :param objectives: the objective functions
        :param get_scalarizer: Create the function for scalarizing the
            objective values. This constructor receives as parameters a `bool`
            which is `True` if and only if all objective functions always
            return integers and `False` otherwise, i.e., if at least one of
            them may return a `float`, the length of the f-vectors, and lists
            with the lower and upper bounds of the objective functions. It can
            use this information to dynamically create and return the most
            efficient scalarization function.
        :param domination: a function reflecting the domination relationship
            between two vectors of objective values. It must obey the contract
            of :meth:`~moptipy.api.mo_problem.MOProblem.f_dominates`, which is
            the same as :func:`moptipy.api.mo_utils.dominates`, to which it
            defaults. `None` overrides nothing.
        """
        if not isinstance(objectives, Iterable):
            raise type_error(objectives, "objectives", Iterable)
        if not callable(get_scalarizer):
            raise type_error(get_scalarizer, "get_scalarizer", call=True)

        lower_bounds: Final[list[int | float]] = []
        upper_bounds: Final[list[int | float]] = []
        calls: Final[list[Callable[[Any], int | float]]] = []
        min_lower_bound: int | float = inf
        max_upper_bound: int | float = -inf

        # Iterate over all objective functions and see whether they are
        # integer-valued and have finite bounds and to collect the bounds.
        always_int: bool = True
        is_int: bool
        lb: int | float
        ub: int | float
        for objective in objectives:
            if not isinstance(objective, Objective):
                raise type_error(objective, "objective[i]", Objective)
            is_int = objective.is_always_integer()
            always_int = always_int and is_int
            calls.append(objective.evaluate)
            lb = objective.lower_bound()
            if isfinite(lb):
                if is_int:
                    if not isinstance(lb, int):
                        raise ValueError(
                            f"if is_always_integer() of objective {objective}"
                            " is True, then lower_bound() must be infinite or"
                            f" int, but is {lb}.")
                else:
                    lb = try_int(lb)
                if lb < min_lower_bound:
                    min_lower_bound = lb
            else:
                min_lower_bound = -inf
            lower_bounds.append(lb)
            ub = objective.upper_bound()
            if isfinite(ub):
                if is_int:
                    if not isinstance(ub, int):
                        raise ValueError(
                            f"if is_always_integer() of objective {objective}"
                            " is True, then upper_bound() must be infinite "
                            f"or int, but is {ub}.")
                else:
                    ub = try_int(ub)
                if ub > max_upper_bound:
                    max_upper_bound = ub
            else:
                max_upper_bound = inf
            if lb >= ub:
                raise ValueError(
                    f"lower_bound()={lb} of objective {objective} must "
                    f"be < than upper_bound()={ub}")
            upper_bounds.append(ub)

        n: Final[int] = len(calls)
        if n <= 0:
            raise ValueError("No objective function found!")

        use_lb: int | float = min_lower_bound
        use_ub: int | float = max_upper_bound
        if always_int:
            if isfinite(min_lower_bound) and isfinite(max_upper_bound):
                use_lb = min(min_lower_bound,
                             min_lower_bound - max_upper_bound)
                use_ub = max(max_upper_bound,
                             max_upper_bound - min_lower_bound)
            else:
                use_lb = -inf
                use_ub = inf

        # Based on the above findings, determine the data type:
        #: The data type of the objective vectors.
        #: If the objectives all always are integers and have known and finite
        #: bounds, then we can use the smallest possible integer type.
        #: This type will be large enough to allow computing "a - b" of any two
        #: objective values "a" and "b" without overflow.
        #: If they are at least integer-valued, we can use the largest integer
        #: type.
        #: If also this is not True, then we just use floating points.
        self.__dtype: Final[np.dtype] = dtype_for_data(
            always_int, use_lb, use_ub)
        #: The dimension of the objective space.
        self.__dimension: Final[int] = n

        #: the creator function for objective vectors
        self.f_create = lambda nn=n, dt=self.__dtype: empty(  # type: ignore
            nn, dt)  # type: ignore

        #: the holder for lower bounds
        self.__lower_bounds: Final[tuple[int | float, ...]] = \
            tuple(lower_bounds)
        #: the holder for upper bounds
        self.__upper_bounds: Final[tuple[int | float, ...]] = \
            tuple(upper_bounds)

        # set up the scalarizer
        self._scalarize: Final[Callable[[np.ndarray], int | float]] \
            = get_scalarizer(always_int, n, lower_bounds, upper_bounds)
        if not callable(self._scalarize):
            raise type_error(self._scalarize, "result of get_scalarizer",
                             call=True)

        # compute the scalarized bounds
        temp: np.ndarray | None = None
        lb = -inf
        if isfinite(min_lower_bound):
            temp = np.array(lower_bounds, dtype=self.__dtype)
            lb = self._scalarize(temp)
            if not isinstance(lb, int | float):
                raise type_error(lb, "computed lower bound", (int, float))
            if (not isfinite(lb)) and (lb > -inf):
                raise ValueError("non-finite computed lower bound "
                                 f"can only be -inf, but is {lb}.")
            lb = try_int(lb)
        #: the lower bound of this scalarization
        self.__lower_bound: Final[int | float] = lb

        ub = inf
        if isfinite(max_upper_bound):
            temp = np.array(upper_bounds, dtype=self.__dtype)
            ub = self._scalarize(temp)
            if not isinstance(ub, int | float):
                raise type_error(ub, "computed upper bound", (int, float))
            if (not isfinite(ub)) and (ub < inf):
                raise ValueError("non-finite computed upper bound "
                                 f"can only be inf, but is {ub}.")
            ub = try_int(ub)
        #: the upper bound of this scalarization
        self.__upper_bound: Final[int | float] = ub

        #: the internal objectives
        self.__calls: Final[tuple[
            Callable[[Any], int | float], ...]] = tuple(calls)
        #: the objective functions
        self._objectives = tuple(objectives)

        #: the internal temporary array
        self._temp: Final[np.ndarray] = self.f_create() \
            if temp is None else temp

        if domination is not None:
            if not callable(domination):
                raise type_error(domination, "domination", call=True)
            self.f_dominates = domination  # type: ignore

    def initialize(self) -> None:
        """Initialize the multi-objective problem."""
        super().initialize()
        for ff in self._objectives:
            ff.initialize()

    def f_dimension(self) -> int:
        """
        Obtain the number of objective functions.

        :returns: the number of objective functions
        """
        return self.__dimension

    def f_dtype(self) -> np.dtype:
        """
        Get the data type used in `f_create`.

        :returns: the data type used by
            :meth:`moptipy.api.mo_problem.MOProblem.f_create`.
        """
        return self.__dtype

    def f_evaluate(self, x, fs: np.ndarray) -> int | float:
        """
        Perform the multi-objective evaluation of a solution.

        :param x: the solution to be evaluated
        :param fs: the array to receive the objective values
        :returns: the scalarized objective values
        """
        for i, o in enumerate(self.__calls):
            fs[i] = o(x)
        return self._scalarize(fs)

    def lower_bound(self) -> float | int:
        """
        Get the lower bound of the scalarization result.

        This function returns a theoretical limit for how good a solution
        could be at best. If no real limit is known, the function returns
        `-inf`.

        :return: the lower bound of the scalarization result
        """
        return self.__lower_bound

    def upper_bound(self) -> float | int:
        """
        Get the upper bound of the scalarization result.

        This function returns a theoretical limit for how bad a solution could
        be at worst. If no real limit is known, the function returns `inf`.

        :return: the upper bound of the scalarization result
        """
        return self.__upper_bound

    def evaluate(self, x) -> float | int:
        """
        Convert the multi-objective problem into a single-objective one.

        This function first evaluates all encapsulated objectives and then
        scalarizes the result.

        :param x: the candidate solution
        :returns: the scalarized objective value
        """
        return self.f_evaluate(x, self._temp)

    def __str__(self) -> str:
        """Get the string representation of this basic scalarization."""
        return "basicMoProblem"

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of this function to the provided destination.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        logger.key_value(KEY_SPACE_NUM_VARS, self.__dimension)
        logger.key_value(KEY_NUMPY_TYPE, numpy_type_to_str(self.__dtype))
        for i, o in enumerate(self._objectives):
            with logger.scope(f"{SCOPE_OBJECTIVE_FUNCTION}{i}") as scope:
                o.log_parameters_to(scope)

    def validate(self, x: np.ndarray) -> None:
        """
        Validate an objective vector.

        :param x: the objective vector
        :raises TypeError: if the string is not an element of this space.
        :raises ValueError: if the shape of the vector is wrong or any of its
            element is not finite.
        """
        super().f_validate(x)

        lb: Final[tuple[int | float, ...]] = self.__lower_bounds
        ub: Final[tuple[int | float, ...]] = self.__upper_bounds
        for i, v in enumerate(x):
            if v < lb[i]:
                raise ValueError(
                    f"encountered {v} at index {i} of {x}, which is below the "
                    f"lower bound {lb[i]} for that position.")
            if v > ub[i]:
                raise ValueError(
                    f"encountered {v} at index {i} of {x}, which is above the "
                    f"upper bound {ub[i]} for that position.")
