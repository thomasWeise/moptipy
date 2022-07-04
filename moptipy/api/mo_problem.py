"""The base classes for multi-objective optimization problems."""
from math import isfinite, inf
from typing import Union, Final, Iterable, Tuple, List, Any, Callable, \
    Optional, cast

import numpy as np
from numpy import empty
from numpy import sum as npsum

from moptipy.api.logging import SCOPE_OBJECTIVE_FUNCTION
from moptipy.api.objective import Objective
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.math import try_int
from moptipy.utils.nputils import dtype_for_data
from moptipy.utils.strings import num_to_str
from moptipy.utils.types import type_error


class MOProblem(Objective):
    """
    The base class for multi-objective optimization problems.

    A multi-objective optimization problem is defined as a set of
    :class:`~moptipy.api.objective.Objective` functions. Each candidate
    solution is evaluated using each of the objectives, i.e., is rated by a
    vector of objective values. This vector is the basis for deciding which
    candidate solutions to keep and which to discard.

    In our implementation, we prescribe that each multi-objective optimization
    problem must also be accompanied by a scalarization function, i.e., a
    function that represents the vector of objective values as a single scalar
    value. The whole multi-objective problem can then be viewed also as a
    single objective function itself. The method
    :meth:`~moptipy.api.mo_problem.MOProblem.evaluate` first evaluates all of
    the objective functions and obtains the vector of objective values. It then
    scalarizes the result into a single scalar quality and returns it.
    Multi-objective algorithms may instead use the method
    :meth:`~moptipy.api.mo_problem.MOProblem.mo_evaluate`, which also allows
    a vector to be passed in which will then be filled with the results of the
    individual objective functions.

    This makes multi-objective optimization with moptipy compatible with
    single-objective optimization. In other words, all optimization methods
    implemented for single-objective processes
    :class:`~moptipy.api.process.Process` will work out-of-the-box with the
    multi-objective version :class:`~moptipy.api.mo_process.MOProcess`.
    """

    def create_f_vector(self) -> np.ndarray:
        """
        Create a vector to receive the objective values.

        :returns: a vector to receive the objective values
        """

    def dimension(self) -> int:
        """
        Obtain the number of objective functions.

        :returns: the number of objective functions
        """

    def mo_evaluate(self, x, fs: np.ndarray) -> Union[int, float]:
        """
        Perform the multi-objective evaluation of a solution.

        This method fills the objective vector `fs` with the results of the
        objective functions evaluated on `x`. It the returns the scalarized
        result, i.e., a single scalar value computed based on all values
        in `fs`.

        :param x: the solution to be evaluated
        :param fs: the array to receive the objective values
        :returns: the scalarization result
        """

    def evaluate(self, x) -> Union[float, int]:
        """
        Evaluate a solution `x` and return its scalarized objective value.

        This method computes all objective values for a given solution and
        then returns the scalarized result. The objective values themselves
        are directly discarted and not used. It makes a multi-objective
        problem compatible with single-objective optimization.

        :param x: the candidate solution
        :returns: the scalarized objective value
        """
        return self.mo_evaluate(x, self.create_f_vector())

    def __str__(self) -> str:
        """Get the string representation of this scalarization."""
        return "scalarization"


class BasicMOProblem(MOProblem):
    """The base class for scalarization."""

    def __init__(self, objectives: Iterable[Objective],
                 get_scalarizer: Callable[
                     [bool, np.dtype, int, Optional[np.ndarray],
                      Optional[np.ndarray]],
                     Callable[[np.ndarray], Union[int, float]]] = None) \
            -> None:
        """
        Create the basic scalarization.

        :param objectives: the objective functions
        :param get_scalarizer: if we overwrite :meth:`scalarize` with a lambda
            or defined function, this function should be used to select it.
            It receives as parameters a `bool` which is `True` if and only if
            all objective functions always return integers and `False`
            otherwise, i.e., if at least one of them may return a `float`; the
            `numpy.dtype` of the objective vectors to be created by
            :meth:`create_f_vector`, the length of the f-vectors, and optional
            arrays with the lower and upper bounds of the objective functions
            that are only specified if all of these bounds are finite.
        """
        super().__init__()
        if not isinstance(objectives, Iterable):
            raise type_error(objectives, "objectives", Iterable)
        if not callable(get_scalarizer):
            raise type_error(get_scalarizer, "get_scalarizer", call=True)

        lower_bounds: Final[List[Union[int, float]]] = []
        upper_bounds: Final[List[Union[int, float]]] = []
        calls: Final[List[Callable[[Any], Union[int, float]]]] = []
        min_lower_bound: Union[int, float] = inf
        max_upper_bound: Union[int, float] = -inf

        # Iterate over all objective functions and see whether they are
        # integer-valued and have finite bounds and to collect the bounds.
        always_int: bool = True
        is_int: bool
        lb: Union[int, float]
        ub: Union[int, float]
        for objective in objectives:
            if not isinstance(objective, Objective):
                raise type_error(objective, "objective[i]", Objective)
            is_int = objective.is_always_integer()
            always_int = always_int and is_int
            calls.append(objective.evaluate)
            lb = objective.lower_bound()
            if isfinite(lb):
                if not is_int:
                    lb = try_int(lb)
                if is_int:
                    if not isinstance(lb, int):
                        raise ValueError(
                            f"if is_always_integer() of objective {objective}"
                            " is True, then lower_bound() must be infinite or"
                            f" int, but is {lb}.")
                if lb < min_lower_bound:
                    min_lower_bound = lb
            else:
                min_lower_bound = -inf
            lower_bounds.append(lb)
            ub = objective.upper_bound()
            if isfinite(ub):
                if not is_int:
                    ub = try_int(ub)
                if is_int:
                    if not isinstance(ub, int):
                        raise ValueError(
                            f"if is_always_integer() of objective {objective}"
                            " is True, then upper_bound() must be infinite "
                            f"or int, but is {ub}.")
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

        # Based on the above findings, determine the data type:
        # If the objectives all always are integers and have known and finite
        # bounds, then we can select smallest possible integer type. If they
        # are at least integer-valued, we can use the largest integer type.
        # If also this is not True, then we just use floating points.
        dtype: Final[np.dtype] = dtype_for_data(always_int, min_lower_bound,
                                                max_upper_bound)

        #: the creator function for objective vectors
        self.create_f_vector = lambda dt=dtype, nn=n: empty(  # type: ignore
            nn, dt)  # type: ignore

        # create the optional array of lower bounds
        lbs: Optional[np.ndarray] = None
        if isfinite(min_lower_bound):
            lbs = np.array(lower_bounds, dtype=dtype)
        ubs: Optional[np.ndarray] = None
        if isfinite(max_upper_bound):
            ubs = np.array(upper_bounds, dtype=dtype)

        # set up the scalarizer
        self.__scalarize: Final[Callable[[np.ndarray], Union[int, float]]] \
            = get_scalarizer(always_int, dtype, n, lbs, ubs)
        if not callable(self.__scalarize):
            raise type_error(self.__scalarize, "result of get_scalarizer",
                             call=True)

        # compute the scalarized bounds
        lb = -inf
        if lbs is not None:
            lb = try_int(self.__scalarize(lbs))
        #: the lower bound of this scalarization
        self.__lower_bound: Final[Union[int, float]] = lb

        ub = inf
        if ubs is not None:
            ub = try_int(self.__scalarize(ubs))
        #: the upper bound of this scalarization
        self.__upper_bound: Final[Union[int, float]] = ub

        #: the internal objectives
        self.__calls: Final[Tuple[
            Callable[[Any], Union[int, float]], ...]] = tuple(calls)

        #: the objective functions
        self.__objectives = tuple(objectives)

        #: the internal temporary array
        self._temp: Final[np.ndarray] = self.create_f_vector()

    def dimension(self) -> int:
        """
        Obtain the number of objective functions.

        :returns: the number of objective functions
        """
        return len(self.__objectives)

    def mo_evaluate(self, x, fs: np.ndarray) -> Union[int, float]:
        """
        Perform the multi-objective evaluation of a solution.

        :param x: the solution to be evaluated
        :param fs: the array to receive the objective values
        :returns: the scalarized objective values
        """
        for i, o in enumerate(self.__calls):
            fs[i] = o(x)
        return self.__scalarize(fs)

    def lower_bound(self) -> Union[float, int]:
        """
        Get the lower bound of the scalarization result.

        This function returns a theoretical limit for how good a solution
        could be at best. If no real limit is known, the function returns
        `-inf`.

        :return: the lower bound of the scalarization result
        """
        return self.__lower_bound

    def upper_bound(self) -> Union[float, int]:
        """
        Get the upper bound of the scalarization result.

        This function returns a theoretical limit for how bad a solution could
        be at worst. If no real limit is known, the function returns `inf`.

        :return: the upper bound of the scalarization result
        """
        return self.__upper_bound

    def evaluate(self, x) -> Union[float, int]:
        """
        Convert the multi-objective problem into a single-objective one.

        This function first evaluates all encapsulated objectives and then
        scalarizes the result.

        :param x: the candidate solution
        :returns: the scalarized objective value
        """
        return self.mo_evaluate(x, self._temp)

    def __str__(self) -> str:
        """Get the string representation of this basic scalarization."""
        return "basicScalarization"

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of this function to the provided destination.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        for i, o in enumerate(self.__objectives):
            with logger.scope(f"{SCOPE_OBJECTIVE_FUNCTION}{i}") as scope:
                o.log_parameters_to(scope)


def _sum_int(a: np.ndarray) -> int:
    """
    Sum up an array and convert the result to an `int` value.

    :param a: the array
    :returns: the sum of the elements in `a` as `int`
    """
    return int(npsum(a))


def _sum_float(a: np.ndarray) -> float:
    """
    Sum up an array and convert the result to a `float` value.

    :param a: the array
    :returns: the sum of the elements in `a` as `float`
    """
    return float(npsum(a))


def _make_sum_scalarizer(always_int: bool,
                         n: int,
                         lower_bounds: Optional[List[Union[int, float]]],
                         upper_bounds: Optional[List[Union[int, float]]],
                         weights: Optional[Tuple[Union[int, float], ...]]) \
        -> Callable[[np.ndarray], Union[int, float]]:
    """
    Creator a weighted sum scalarization function.

    If `weights` is `None`, we will just use the plain summation function from
    numpy and convert its result to `int` if `always_int` is `True` and to
    `float` otherwise.
    If `weights` is a tuple of weights, then we will convert it to a numpy
    `ndarray`. If `always_int` is `True` and all weights are integers, and the
    lower and upper bound of the objectives are known, we try to pick the
    smallest integer data type for this array big enough to hold both the
    total lower and total upper bound. If at least the lower bounds are >= 0,
    then we pick the largest unsigned integer data type. If the bounds are
    unknown, then we pick the largest signed integer type. If any weight is
    not an integer and `always_int` is `False`, we use a default floating
    point weight array.

    This should yield the overall fastest, most precise, and most memory
    efficient way to compute a weighted sum scalarization.

    :param always_int: will all objectives always be integer
    :param n: the number of objectives
    :param lower_bounds: the optional lower bounds
    :param upper_bounds: the optional upper bounds
    :param weights: the optional array of weights, `None` if all weights
        are `1`.
    :returns: the scalarization function
    """
    min_sum: Union[int, float]
    if lower_bounds is None:
        min_sum = 0
    elif not isinstance(lower_bounds, List):
        raise type_error(lower_bounds, "lower_bounds", List)
    elif len(lower_bounds) != n:
        raise ValueError(
            f"there should be {n} values in lower_bounds={lower_bounds}")
    else:
        min_sum = -inf

    max_sum: Union[int, float]
    if upper_bounds is None:
        max_sum = 0
    elif not isinstance(upper_bounds, List):
        raise type_error(upper_bounds, "upper_bounds", List)
    elif len(upper_bounds) != n:
        raise ValueError(
            f"there should be {n} values in upper_bounds={lower_bounds}")
    else:
        max_sum = inf

    if weights is None:
        return _sum_int if always_int else _sum_float
    if not isinstance(weights, tuple):
        raise type_error(weights, "weights", tuple)
    if len(weights) != n:
        raise ValueError(
            f"there should be {n} values in weights={lower_bounds}")

    if not isinstance(always_int, bool):
        raise type_error(always_int, "always_int", bool)

    min_weight: Union[int, float] = inf
    max_weight: Union[int, float] = -inf
    everything_is_int: bool = always_int

    for i, weight in enumerate(weights):
        if weight < 0:
            raise ValueError("no weight can be <0, but encountered "
                             f"{weight} in {weights}.")

        if not isinstance(weight, int):
            everything_is_int = False
            if not isfinite(weight):
                raise ValueError("weight must be finite, but "
                                 f"encountered {weight} in {weights}.")
            min_sum = -inf
            max_sum = inf
            break

        if weight < min_weight:
            min_weight = weight
        if weight > max_weight:
            max_weight = weight

        if lower_bounds is not None:
            min_sum += weight * lower_bounds[i]
        if upper_bounds is not None:
            max_sum += weight * upper_bounds[i]

    if min_sum >= max_sum:
        raise ValueError(
            f"weighted sum minimum={min_sum} >= maximum={max_sum}?")

    # re-check for plain summation
    if 1 <= min_weight <= max_weight <= 1:
        return _sum_int if always_int else _sum_float
    use_weights: Final[np.ndarray] = np.array(weights, dtype_for_data(
        everything_is_int, min_sum, max_sum))
    if everything_is_int:
        return cast(Callable[[np.ndarray], Union[int, float]],
                    lambda a, w=use_weights: int(a * w))
    return cast(Callable[[np.ndarray], Union[int, float]],
                lambda a, w=use_weights: float(a * w))


class WeightedSumScalarization(BasicMOProblem):
    """Scalarize objective values by computing their weighted sum."""

    def __init__(self, objectives: Iterable[Objective],
                 weights: Optional[Iterable[Union[int, float]]] = None) \
            -> None:
        """
        Create the sum-based scalarization.

        :param objectives: the objectives
        :param weights: the weights of the objective values, or `None` if all
            weights are `1`.
        """
        use_weights: Optional[Tuple[Union[int, float], ...]] \
            = None if weights is None else tuple(try_int(w) for w in weights)

        super().__init__(
            objectives,
            cast(Callable[[bool, np.dtype, int, Optional[np.ndarray],
                           Optional[np.ndarray]], Callable[
                [np.ndarray], Union[int, float]]],
                lambda ai, d, n, lb, ub, uw=use_weights:
                _make_sum_scalarizer(ai, n, lb, ub, uw)))

        #: the internal weights
        self.__weights: Final[Optional[Tuple[Union[int, float], ...]]] = \
            use_weights

    def __str__(self):
        """
        Get the string representation of the weighted sum scalarization.

        :returns: `"weightedSum"`
        """
        return "weightedSum"

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of this function to the provided destination.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        logger.key_value("weights", "None" if self.__weights is None else
                         ";".join([num_to_str(w) for w in self.__weights]))
