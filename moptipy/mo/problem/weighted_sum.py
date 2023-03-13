"""
The weighted sum scalarization of multi-objective problems.

Here we provide two simple methods to scalarize multi-objective problems by
using weights, namely

- :class:`~moptipy.mo.problem.weighted_sum.WeightedSum`, a sum with arbitrary,
  user-defined weights of the objective values
- :class:`~moptipy.mo.problem.weighted_sum.Prioritize`, a weighted sum of the
  objective values where the weights are automatically determined such that
  the first objective function is prioritized over the second one, the second
  one over the third, and so on.
"""
from math import inf, isfinite
from typing import Any, Callable, Final, Iterable, cast

import numpy as np
from numpy import sum as npsum

from moptipy.api.mo_utils import dominates
from moptipy.api.objective import Objective
from moptipy.mo.problem.basic_mo_problem import BasicMOProblem
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.math import try_int
from moptipy.utils.nputils import dtype_for_data
from moptipy.utils.strings import num_to_str
from moptipy.utils.types import type_error


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


class BasicWeightedSum(BasicMOProblem):
    """
    Base class for scalarizing objective values by a weighted sum.

    This class brings the basic tools to scalarize vectors of objective
    values by computing weighted sums. This class should not be used
    directly. Instead, use its sub-classes
    :class:`~moptipy.mo.problem.weighted_sum.WeightedSum` and
    :class:`~moptipy.mo.problem.weighted_sum.Prioritize`.
    """

    def __init__(self, objectives: Iterable[Objective],
                 get_scalarizer: Callable[
                     [bool, int, list[int | float],
                      list[int | float], Callable[
                          [None | np.dtype | tuple[int | float, ...]], None]],
                     Callable[[np.ndarray], int | float]],
                 domination: Callable[[np.ndarray, np.ndarray], int] | None
                 = dominates) -> None:
        """
        Create the sum-based scalarization.

        :param objectives: the objectives
        :param domination: a function reflecting the domination relationship
            between two vectors of objective values. It must obey the contract
            of :meth:`~moptipy.api.mo_problem.MOProblem.f_dominates`, which is
            the same as :func:`moptipy.api.mo_utils.dominates`, to which it
            defaults. `None` overrides nothing.
        """
        holder: list[Any] = []
        super().__init__(
            objectives,
            cast(Callable[[bool, int, list[int | float],
                           list[int | float]], Callable[
                [np.ndarray], int | float]],
                lambda ai, n, lb, ub, fwd=holder.append:
                get_scalarizer(ai, n, lb, ub, fwd)),
            domination)
        if len(holder) != 2:
            raise ValueError(
                f"need weights and weights dtype, but got {holder}.")
        #: the internal weights
        self.weights: Final[tuple[int | float, ...] | None] = \
            cast(tuple[int | float, ...] | None, holder[0])
        if self.weights is not None:
            if not isinstance(self.weights, tuple):
                raise type_error(self.weights, "weights", [tuple, None])
            if len(self.weights) != self.f_dimension():
                raise ValueError(
                    f"length of weights {self.weights} is not "
                    f"f_dimension={self.f_dimension()}.")
        #: the internal weights dtype
        self.__weights_dtype: Final[np.dtype | None] = \
            cast(np.dtype | None, holder[1])
        if (self.__weights_dtype is not None) \
                and (not isinstance(self.__weights_dtype, np.dtype)):
            raise type_error(
                self.__weights_dtype, "weights_dtype", np.dtype)

    def __str__(self):
        """
        Get the string representation of the weighted sum scalarization.

        :returns: `"weightedSumBase"`
        """
        return "weightedSumBase"

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of this function to the provided destination.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)

        weights: tuple[int | float, ...] | None = self.weights
        logger.key_value("weights", ";".join(
            (["1"] * self.f_dimension()) if weights is None else
            [num_to_str(w) for w in weights]))
        logger.key_value("weightsDtype",
                         "None" if self.__weights_dtype is None
                         else self.__weights_dtype.char)


def _make_sum_scalarizer(
        always_int: bool, n: int,
        lower_bounds: list[int | float], upper_bounds: list[int | float],
        weights: tuple[int | float, ...] | None,
        callback: Callable[[None | np.dtype | tuple[
            int | float, ...]], None]) -> Callable[[np.ndarray], int | float]:
    """
    Create a weighted sum scalarization function.

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
    :param callback: the callback function to receive the weights
    :returns: the scalarization function
    """
    if not isinstance(lower_bounds, list):
        raise type_error(lower_bounds, "lower_bounds", list)
    if len(lower_bounds) != n:
        raise ValueError(
            f"there should be {n} values in lower_bounds={lower_bounds}")

    if not isinstance(upper_bounds, list):
        raise type_error(upper_bounds, "upper_bounds", list)
    if len(upper_bounds) != n:
        raise ValueError(
            f"there should be {n} values in upper_bounds={lower_bounds}")

    if weights is None:
        callback(None)
        callback(None)
        return _sum_int if always_int else _sum_float
    if not isinstance(weights, tuple):
        raise type_error(weights, "weights", tuple)
    if len(weights) != n:
        raise ValueError(
            f"there should be {n} values in weights={lower_bounds}")
    if not isinstance(always_int, bool):
        raise type_error(always_int, "always_int", bool)

    min_sum: int | float = 0
    max_sum: int | float = 0
    min_weight: int | float = inf
    max_weight: int | float = -inf
    everything_is_int: bool = always_int

    for i, weight in enumerate(weights):
        if weight <= 0:
            raise ValueError("no weight can be <=0, but encountered "
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
        callback(None)
        callback(None)
        return _sum_int if always_int else _sum_float

    dtype: Final[np.dtype] = dtype_for_data(
        everything_is_int, min_sum, max_sum)
    use_weights: Final[np.ndarray] = np.array(weights, dtype)

    callback(weights)
    callback(dtype)

    if everything_is_int:
        return cast(Callable[[np.ndarray], int | float],
                    lambda a, w=use_weights: int(npsum(a * w)))
    return cast(Callable[[np.ndarray], int | float],
                lambda a, w=use_weights: float(npsum(a * w)))


class WeightedSum(BasicWeightedSum):
    """Scalarize objective values by computing their weighted sum."""

    def __init__(self, objectives: Iterable[Objective],
                 weights: Iterable[int | float] | None = None,
                 domination: Callable[[np.ndarray, np.ndarray], int] | None
                 = dominates) -> None:
        """
        Create the sum-based scalarization.

        :param objectives: the objectives
        :param weights: the weights of the objective values, or `None` if all
            weights are `1`.
        :param domination: a function reflecting the domination relationship
            between two vectors of objective values. It must obey the contract
            of :meth:`~moptipy.api.mo_problem.MOProblem.f_dominates`, which is
            the same as :func:`moptipy.api.mo_utils.dominates`, to which it
            defaults. `None` overrides nothing.
        """
        use_weights: tuple[int | float, ...] | None \
            = None if weights is None else tuple(try_int(w) for w in weights)

        super().__init__(
            objectives,
            cast(Callable[
                 [bool, int, list[int | float],
                  list[int | float], Callable[
                      [None | np.dtype | tuple[int | float, ...]], None]],
                 Callable[[np.ndarray], int | float]],
                 lambda ai, n, lb, ub, cb, uw=use_weights:
                 _make_sum_scalarizer(ai, n, lb, ub, uw, cb)),
            domination)

    def __str__(self):
        """
        Get the string representation of the weighted sum scalarization.

        :returns: `"weightedSum"`
        """
        return "weightedSum" if self.f_dominates is dominates \
            else "weightedSumWithDominationFunc"


def _prioritize(
        always_int: bool, n: int,
        lower_bounds: list[int | float], upper_bounds: list[int | float],
        callback: Callable[[None | np.dtype | tuple[
            int | float, ...]], None]) -> Callable[[np.ndarray], int | float]:
    """
    Create a weighted-sum based prioritization of the objective functions.

    If all objective functions are integers and have upper and lower bounds,
    we can use integer weights to create a prioritization such that gaining
    one unit of the first objective function is always more important than any
    improvement of the second objective, that gaining one unit of the second
    objective always outweighs all possible gains in terms of the third one,
    and so on.

    :param always_int: will all objectives always be integer
    :param n: the number of objectives
    :param lower_bounds: the optional lower bounds
    :param upper_bounds: the optional upper bounds
    :param callback: the callback function to receive the weights
    :returns: the scalarization function
    """
    if n == 1:
        return _make_sum_scalarizer(always_int, n, lower_bounds,
                                    upper_bounds, None, callback)
    if not always_int:
        raise ValueError("priority-based weighting is only possible for "
                         "integer-valued objectives")
    weights: list[int] = [1]
    weight: int = 1
    for i in range(n - 1, 0, -1):
        lb: int | float = lower_bounds[i]
        if not isinstance(lb, int | float):
            raise type_error(lb, f"lower_bound[{i}]", (int, float))
        if not isfinite(lb):
            raise ValueError(f"lower_bound[{i}]={lb}, but must be finite")
        if not isinstance(lb, int):
            raise type_error(lb, f"finite lower_bound[{i}]", int)
        ub: int | float = upper_bounds[i]
        if not isinstance(ub, int | float):
            raise type_error(ub, f"upper_bound[{i}]", (int, float))
        if not isfinite(ub):
            raise ValueError(f"upper_bound[{i}]={ub}, but must be finite")
        if not isinstance(ub, int):
            raise type_error(ub, f"finite upper_bound[{i}]", int)
        weight *= (1 + cast(int, ub) - min(0, cast(int, lb)))
        weights.append(weight)

    weights.reverse()
    return _make_sum_scalarizer(always_int, n, lower_bounds, upper_bounds,
                                tuple(weights), callback)


class Prioritize(BasicWeightedSum):
    """Prioritize the first objective over the second and so on."""

    def __init__(self, objectives: Iterable[Objective],
                 domination: Callable[[np.ndarray, np.ndarray], int] | None
                 = dominates) -> None:
        """
        Create the sum-based prioritization.

        :param objectives: the objectives
        :param domination: a function reflecting the domination relationship
            between two vectors of objective values. It must obey the contract
            of :meth:`~moptipy.api.mo_problem.MOProblem.f_dominates`, which is
            the same as :func:`moptipy.api.mo_utils.dominates`, to which it
            defaults. `None` overrides nothing.
        """
        super().__init__(objectives, _prioritize, domination)

    def __str__(self):
        """
        Get the name of the weighted sum-based prioritization.

        :returns: `"weightBasedPrioritization"`
        """
        return "weightBasedPrioritization" if self.f_dominates is dominates \
            else "weightBasedPrioritizationWithDominationFunc"
