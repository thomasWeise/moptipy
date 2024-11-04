"""
A temperature schedule as needed by Simulated Annealing.

The Simulated Annealing algorithm implemented in
:mod:`~moptipy.algorithms.so.simulated_annealing` performs a local search that
always accepts a non-worsening move, i.e., a solution which is not worse than
the currently maintained one. However, it will also *sometimes* accept one
that is worse. The probability of doing so depends on how much worse that
solution is and on the current *temperature* of the algorithm. The higher the
temperature, the higher the acceptance probability. The temperature changes
over time according to the
:class:`~moptipy.algorithms.modules.temperature_schedule.TemperatureSchedule`.

The temperature schedule receives an iteration index `tau` as input and
returns the current temperature via :meth:`~moptipy.algorithms.modules.\
temperature_schedule.TemperatureSchedule.temperature`. Notice that `tau` is
zero-based for simplicity reason, meanings that the first objective function
evaluation is at index `0`.
"""

from math import e, isfinite, log
from typing import Final

from pycommons.math.int_math import try_int
from pycommons.strings.enforce import enforce_non_empty_str_without_ws
from pycommons.types import check_int_range, type_error

from moptipy.api.component import Component
from moptipy.api.objective import Objective, check_objective
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.strings import num_to_str_for_name


# start schedule
class TemperatureSchedule(Component):
    """The base class for temperature schedules."""

    def __init__(self, t0: float) -> None:
        # end schedule
        """
        Initialize the temperature schedule.

        :param t0: the starting temperature, must be > 0
        """
        super().__init__()
        if not isinstance(t0, float):
            raise type_error(t0, "t0", float)
        if (not isfinite(t0)) or (t0 <= 0.0):
            raise ValueError(f"t0 must be >0, cannot be {t0}.")
# start schedule
        #: the starting temperature
        self.t0: Final[float] = t0

    def temperature(self, tau: int) -> float:
        """
        Compute the temperature at iteration `tau`.

        :param tau: the iteration index, starting with `0` at the first
            comparison of two solutions, at which point the starting
            temperature :attr:`~TemperatureSchedule.t0` should be returned
        :returns: the temperature
        """
# end schedule

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log all parameters of this temperature schedule as key-value pairs.

        :param logger: the logger for the parameters

        >>> from moptipy.utils.logger import InMemoryLogger
        >>> with InMemoryLogger() as l:
        ...     with l.key_values("C") as kv:
        ...         TemperatureSchedule(0.1).log_parameters_to(kv)
        ...     text = l.get_log()
        >>> text[1]
        'name: TemperatureSchedule'
        >>> text[3]
        'T0: 0.1'
        >>> len(text)
        6
        """
        super().log_parameters_to(logger)
        logger.key_value("T0", self.t0)


# start exponential
class ExponentialSchedule(TemperatureSchedule):
    """
    The exponential temperature schedule.

    The current temperature is computed as `t0 * (1 - epsilon) ** tau`.

    >>> ex = ExponentialSchedule(10.0, 0.05)
    >>> print(f"{ex.t0} - {ex.epsilon}")
    10.0 - 0.05
    >>> ex.temperature(0)
    10.0
    >>> ex.temperature(1)
    9.5
    >>> ex.temperature(2)
    9.025
    >>> ex.temperature(1_000_000_000_000_000_000)
    0.0
    """

    def __init__(self, t0: float, epsilon: float) -> None:
        """
        Initialize the exponential temperature schedule.

        :param t0: the starting temperature, must be > 0
        :param epsilon: the epsilon parameter of the schedule, in (0, 1)
        """
        super().__init__(t0)
# end exponential
        if not isinstance(epsilon, float):
            raise type_error(epsilon, "epsilon", float)
        if (not isfinite(epsilon)) or (not (0.0 < epsilon < 1.0)):
            raise ValueError(
                f"epsilon cannot be {epsilon}, must be in (0,1).")
# start exponential
        #: the epsilon parameter of the exponential schedule
        self.epsilon: Final[float] = epsilon
        #: the value used as basis for the exponent
        self.__one_minus_epsilon: Final[float] = 1.0 - epsilon
# end exponential
        if not (0.0 < self.__one_minus_epsilon < 1.0):
            raise ValueError(
                f"epsilon cannot be {epsilon}, because 1-epsilon must be in "
                f"(0, 1) but is {self.__one_minus_epsilon}.")
# start exponential

    def temperature(self, tau: int) -> float:
        """
        Compute the temperature at iteration `tau`.

        :param tau: the iteration index, starting with `0` at the first
            comparison of two solutions, at which point the starting
            temperature :attr:`~TemperatureSchedule.t0` should be returned
        :returns: the temperature

        >>> s = ExponentialSchedule(100.0, 0.5)
        >>> s.temperature(0)
        100.0
        >>> s.temperature(1)
        50.0
        >>> s.temperature(10)
        0.09765625
        """
        return self.t0 * (self.__one_minus_epsilon ** tau)
# end exponential

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log all parameters of the exponential temperature schedule.

        :param logger: the logger for the parameters

        >>> from moptipy.utils.logger import InMemoryLogger
        >>> with InMemoryLogger() as l:
        ...     with l.key_values("C") as kv:
        ...         ExponentialSchedule(0.2, 0.6).log_parameters_to(kv)
        ...     text = l.get_log()
        >>> text[1]
        'name: exp0d2_0d6'
        >>> text[3]
        'T0: 0.2'
        >>> text[5]
        'e: 0.6'
        >>> len(text)
        8
        """
        super().log_parameters_to(logger)
        logger.key_value("e", self.epsilon)

    def __str__(self) -> str:
        """
        Get the string representation of the exponential temperature schedule.

        :returns: the name of this schedule

        >>> ExponentialSchedule(100.5, 0.3)
        exp100d5_0d3
        """
        return (f"exp{num_to_str_for_name(self.t0)}_"
                f"{num_to_str_for_name(self.epsilon)}")


# start logarithmic
class LogarithmicSchedule(TemperatureSchedule):
    """
    The logarithmic temperature schedule.

    The temperature is computed as `t0 / log(e + (tau * epsilon))`.

    >>> lg = LogarithmicSchedule(10.0, 0.1)
    >>> print(f"{lg.t0} - {lg.epsilon}")
    10.0 - 0.1
    >>> lg.temperature(0)
    10.0
    >>> lg.temperature(1)
    9.651322627630812
    >>> lg.temperature(1_000_000_000_000_000_000_000_000_000_000_000_000_000)
    0.11428802155348732
    """

    def __init__(self, t0: float, epsilon: float) -> None:
        """
        Initialize the logarithmic temperature schedule.

        :param t0: the starting temperature, must be > 0
        :param epsilon: the epsilon parameter of the schedule, is > 0
        """
        super().__init__(t0)
# end logarithmic
        if not isinstance(epsilon, float):
            raise type_error(epsilon, "epsilon", float)
        if (not isfinite(epsilon)) or (epsilon <= 0.0):
            raise ValueError(
                f"epsilon cannot be {epsilon}, must be > 0.")
# start logarithmic
        #: the epsilon parameter of the logarithmic schedule
        self.epsilon: Final[float] = epsilon

    def temperature(self, tau: int) -> float:
        """
        Compute the temperature at iteration `tau`.

        :param tau: the iteration index, starting with `0` at the first
            comparison of two solutions, at which point the starting
            temperature :attr:`~TemperatureSchedule.t0` should be returned
        :returns: the temperature

        >>> s = LogarithmicSchedule(100.0, 0.5)
        >>> s.temperature(0)
        100.0
        >>> s.temperature(1)
        85.55435113150568
        >>> s.temperature(10)
        48.93345190925178
        """
        return self.t0 / log(e + (tau * self.epsilon))
# end logarithmic

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log all parameters of the logarithmic temperature schedule.

        :param logger: the logger for the parameters

        >>> from moptipy.utils.logger import InMemoryLogger
        >>> with InMemoryLogger() as l:
        ...     with l.key_values("C") as kv:
        ...         LogarithmicSchedule(0.2, 0.6).log_parameters_to(kv)
        ...     text = l.get_log()
        >>> text[1]
        'name: ln0d2_0d6'
        >>> text[3]
        'T0: 0.2'
        >>> text[5]
        'e: 0.6'
        >>> len(text)
        8
        """
        super().log_parameters_to(logger)
        logger.key_value("e", self.epsilon)

    def __str__(self) -> str:
        """
        Get the string representation of the logarithmic temperature schedule.

        :returns: the name of this schedule

        >>> LogarithmicSchedule(100.5, 0.3)
        ln100d5_0d3
        """
        return (f"ln{num_to_str_for_name(self.t0)}_"
                f"{num_to_str_for_name(self.epsilon)}")


class ExponentialScheduleBasedOnBounds(ExponentialSchedule):
    """
    An exponential schedule configured based on the objective's range.

    This exponential schedule takes an objective function as parameter.
    It uses the lower and the upper bound of this function, `LB` and `UB`,
    to select a start and end temperature based on the provided fractions.
    Here, we set `W = lb_sum_weight * LB + ub_sum_weight * UB`.
    If we set `lb_sum_weight = -1` and `ub_sum_weight = 1`, then `W` will be
    the range of the objective function.
    If we set `lb_sum_weight = 1` and `ub_sum_weight = 0`, then we base the
    temperature setup entirely on the lower bound.
    If we set `lb_sum_weight = 0` and `ub_sum_weight = 1`, then we base the
    temperature setup entirely on the upper bound.
    Roughly, the start temperature will be `W * start_range_frac` and
    the end temperature, to be reached after `n_steps` FEs, will be
    `W * end_range_frac`.
    Since sometimes the upper and lower bound may be excessivly large, we
    can provide limits for `W` in form of `min_bound_sum` and `max_bound_sum`.
    This will then override any other computation.
    Notice that it is expected that `tau == 0` when the temperature function
    is first called. It is expected that `tau == n_range - 1` when it is
    called for the last time.

    >>> from moptipy.examples.bitstrings.onemax import OneMax
    >>> es = ExponentialScheduleBasedOnBounds(
    ...     OneMax(10), -1, 1, 0.01, 0.0001, 10**8)
    >>> es.temperature(0)
    0.1
    >>> es.temperature(1)
    0.09999999539482989
    >>> es.temperature(10**8 - 1)
    0.0010000000029841878
    >>> es.temperature(10**8)
    0.0009999999569324865

    >>> es = ExponentialScheduleBasedOnBounds(
    ...         OneMax(10), -1, 1, 0.01, 0.0001, 10**8, max_bound_sum=5)
    >>> es.temperature(0)
    0.05
    >>> es.temperature(1)
    0.04999999769741494
    >>> es.temperature(10**8 - 1)
    0.0005000000014920939
    >>> es.temperature(10**8)
    0.0004999999784662432

    >>> try:
    ...     ExponentialScheduleBasedOnBounds(1, 0.01, 0.0001, 10**8)
    ... except TypeError as te:
    ...     print(te)
    objective function should be an instance of moptipy.api.objective.\
Objective but is int, namely 1.

    >>> try:
    ...     ExponentialScheduleBasedOnBounds(
    ...         OneMax(10), -1, 1, -1.0, 0.0001, 10**8)
    ... except ValueError as ve:
    ...     print(ve)
    Invalid bound sum factors [-1.0, 0.0001].

    >>> try:
    ...     ExponentialScheduleBasedOnBounds(
    ...         OneMax(10), -1, 1, 0.9, 0.0001, 1)
    ... except ValueError as ve:
    ...     print(ve)
    n_steps=1 is invalid, must be in 2..1000000000000000.
    """

    def __init__(self, f: Objective,
                 lb_sum_weight: int | float = -1,
                 ub_sum_weight: int | float = 1,
                 start_factor: float = 1e-3,
                 end_factor: float = 1e-7,
                 n_steps: int = 1_000_000,
                 min_bound_sum: int | float = 1e-20,
                 max_bound_sum: int | float = 1e20) -> None:
        """
        Initialize the range-based exponential schedule.

        :param f: the objective function whose range we will use
        :param lb_sum_weight: the weight of the lower bound in the bound sum
        :param ub_sum_weight: the weight of the upper bound in the bound sum
        :parma start_factor: the factor multiplied with the bound sum to get
            the starting temperature
        :parm end_factor: the factor multiplied with the bound sum to get
            the end temperature
        :param n_steps: the number of steps until the end range should be
            reached
        :param min_bound_sum: a lower limit for the weighted sum of the bounds
        :param max_bound_sum: an upper limit for the weighted sum of the bounds
        """
        f = check_objective(f)
        if not isinstance(lb_sum_weight, int | float):
            raise type_error(lb_sum_weight, "lb_sum_weight", (int, float))
        if not isinstance(ub_sum_weight, int | float):
            raise type_error(ub_sum_weight, "ub_sum_weight", (int, float))
        if not isinstance(start_factor, float):
            raise type_error(start_factor, "start_factor", float)
        if not isinstance(end_factor, float):
            raise type_error(end_factor, "end_factor", float)
        if not isinstance(min_bound_sum, int | float):
            raise type_error(min_bound_sum, "min_bound_sum", (int, float))
        if not isinstance(max_bound_sum, int | float):
            raise type_error(max_bound_sum, "max_bound_sum", (int, float))

        if not (isfinite(min_bound_sum) and isfinite(max_bound_sum) and (
                0 < min_bound_sum < max_bound_sum)):
            raise ValueError(f"Invalid bound sum limits [{min_bound_sum}"
                             f", {max_bound_sum}].")
        if not (isfinite(start_factor) and isfinite(end_factor) and (
                0.0 < end_factor < start_factor < 1e50)):
            raise ValueError(f"Invalid bound sum factors [{start_factor}"
                             f", {end_factor}].")
        if not (isfinite(lb_sum_weight) and isfinite(ub_sum_weight)):
            raise ValueError(f"Invalid bound sum weights [{lb_sum_weight}"
                             f", {ub_sum_weight}].")
        #: the number of steps that we will perform until reaching the end
        #: range fraction temperature
        self.__n_steps: Final[int] = check_int_range(
            n_steps, "n_steps", 2, 1_000_000_000_000_000)

        lb_sum_weight = try_int(lb_sum_weight)
        #: the sum weight for the lower bound
        self.__lb_sum_weight: Final[int | float] = lb_sum_weight
        ub_sum_weight = try_int(ub_sum_weight)
        #: the sum weight for the upper bound
        self.__ub_sum_weight: Final[int | float] = ub_sum_weight
        #: the start temperature bound sum factor
        self.__start_factor: Final[float] = start_factor
        #: the end temperature bound sum factor
        self.__end_factor: Final[float] = end_factor
        min_bound_sum = try_int(min_bound_sum)
        #: the minimum value for the bound sum
        self.__min_bound_sum: Final[int | float] = min_bound_sum
        max_bound_sum = try_int(max_bound_sum)
        #: the maximum value for the bound sum
        self.__max_bound_sum: Final[int | float] = max_bound_sum

        #: the name of the objective function used
        self.__used_objective: Final[str] = enforce_non_empty_str_without_ws(
            str(f))
        flb: Final[float | int] = f.lower_bound()
        fub: Final[float | int] = f.upper_bound()
        if flb > fub:
            raise ValueError(
                f"Objective function lower bound {flb} > upper bound {fub}?")

        #: the lower bound of the objective value
        self.__f_lower_bound: Final[int | float] = flb
        #: the upper bound for the objective value
        self.__f_upper_bound: Final[int | float] = fub

        bound_sum: Final[float | int] = try_int(max(min_bound_sum, min(
            max_bound_sum,
            (flb * lb_sum_weight if lb_sum_weight != 0 else 0) + (
                fub * ub_sum_weight if ub_sum_weight != 0 else 0))))
        if not (isfinite(bound_sum) and (
                min_bound_sum <= bound_sum <= max_bound_sum)):
            raise ValueError(
                f"Invalid bound sum {bound_sum} resulting from bounds [{flb}"
                f", {fub}] and weights {lb_sum_weight}, {ub_sum_weight}.")
        #: the bound sum
        self.__f_bound_sum: Final[int | float] = bound_sum

        t0: Final[float] = start_factor * bound_sum
        te: Final[float] = end_factor * bound_sum
        if not (isfinite(t0) and isfinite(te) and (0 < te < t0 < 1e100)):
            raise ValueError(
                f"Invalid setup {start_factor}, {end_factor}, "
                f"{bound_sum} leading to temperatures {t0}, {te}.")
        #: the end temperature
        self.__te: Final[float] = te

        epsilon: Final[float] = 1 - (te / t0) ** (1 / (n_steps - 1))
        if not (isfinite(epsilon) and (0 < epsilon < 1) and (
                0 < (1 - epsilon) < 1)):
            raise ValueError(
                f"Invalid computed epsilon {epsilon} resulting from setup "
                f"{start_factor}, {end_factor}, {bound_sum} leading "
                f"to temperatures {t0}, {te}.")
        super().__init__(t0, epsilon)

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log all parameters of the configured exponential temperature schedule.

        :param logger: the logger for the parameters

        >>> from moptipy.utils.logger import InMemoryLogger
        >>> from moptipy.examples.bitstrings.onemax import OneMax
        >>> with InMemoryLogger() as l:
        ...     with l.key_values("C") as kv:
        ...         ExponentialScheduleBasedOnBounds(
        ...             OneMax(10), -1, 1, 0.01, 0.0001).log_parameters_to(kv)
        ...     text = l.get_log()
        >>> text[1]
        'name: expRm1_1_0d01_0d0001_1em20_1e20'
        >>> text[3]
        'T0: 0.1'
        >>> text[5]
        'e: 4.6051641873212645e-6'
        >>> text[7]
        'nSteps: 1000000'
        >>> text[8]
        'lbSumWeight: -1'
        >>> text[9]
        'ubSumWeight: 1'
        >>> len(text)
        25
        """
        super().log_parameters_to(logger)
        logger.key_value("nSteps", self.__n_steps)
        logger.key_value("lbSumWeight", self.__lb_sum_weight)
        logger.key_value("ubSumWeight", self.__ub_sum_weight)
        logger.key_value("startFactor", self.__start_factor)
        logger.key_value("endFactor", self.__end_factor)
        logger.key_value("minBoundSum", self.__min_bound_sum)
        logger.key_value("maxBoundSum", self.__max_bound_sum)
        logger.key_value("f", self.__used_objective)
        logger.key_value("fLB", self.__f_lower_bound)
        logger.key_value("fUB", self.__f_upper_bound)
        logger.key_value("boundSum", self.__f_bound_sum)
        logger.key_value("Tend", self.__te)

    def __str__(self) -> str:
        """
        Get the string representation of the configured exponential schedule.

        :returns: the name of this schedule

        >>> from moptipy.examples.bitstrings.onemax import OneMax
        >>> ExponentialScheduleBasedOnBounds(OneMax(10), -1, 1, 0.01, 0.0001)
        expRm1_1_0d01_0d0001_1em20_1e20
        """
        return "expR" + "_".join(map(num_to_str_for_name, (
            self.__lb_sum_weight, self.__ub_sum_weight, self.__start_factor,
            self.__end_factor, self.__min_bound_sum, self.__max_bound_sum)))
