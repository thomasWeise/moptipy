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

from math import e, inf, isfinite, log, nextafter
from typing import Final

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


#: the default maximum range
_DEFAULT_MAX_RANGE: Final[float] = inf

#: the default minimum range
_DEFAULT_MIN_RANGE: Final[float] = nextafter(0.0, _DEFAULT_MAX_RANGE)


class ExponentialScheduleBasedOnRange(ExponentialSchedule):
    """
    An exponential schedule configured based on the objective's range.

    This exponential schedule takes an objective function as parameter.
    It uses the lower and the upper bound of this function, `LB` and `UB`,
    to select a start and end temperature based on the provided fractions.
    Here, we set `R = UB - LB`.
    Roughly, the start temperature will be `R * start_range_frac` and
    the end temperature, to be reached after `n_steps` FEs, will be
    `R * end_range_frac`.
    If one of `UB` or `LB` is not provided, we use `R = max(1, abs(other))`.
    If neither is provided, we set `R = 1`.
    Since sometimes the upper and lower bound may be excessivly large, we
    can provide limits for `R` in form of `min_range` and `max_range`.
    This will then override any other computation.
    Notice that it is expected that `tau == 0` when the temperature function
    is first called. It is expected that `tau == n_range - 1` when it is
    called for the last time.

    >>> from moptipy.examples.bitstrings.onemax import OneMax
    >>> es = ExponentialScheduleBasedOnRange(OneMax(10), 0.01, 0.0001, 10**8)
    >>> es.temperature(0)
    0.1
    >>> es.temperature(1)
    0.09999999539482989
    >>> es.temperature(10**8 - 1)
    0.0010000000029841878
    >>> es.temperature(10**8)
    0.0009999999569324865

    >>> es = ExponentialScheduleBasedOnRange(
    ...         OneMax(10), 0.01, 0.0001, 10**8, max_range=5)
    >>> es.temperature(0)
    0.05
    >>> es.temperature(1)
    0.04999999769741494
    >>> es.temperature(10**8 - 1)
    0.0005000000014920939
    >>> es.temperature(10**8)
    0.0004999999784662432

    >>> try:
    ...     ExponentialScheduleBasedOnRange(1, 0.01, 0.0001, 10**8)
    ... except TypeError as te:
    ...     print(te)
    objective function should be an instance of moptipy.api.objective.\
Objective but is int, namely 1.

    >>> try:
    ...     ExponentialScheduleBasedOnRange(OneMax(10), 12.0, 0.0001, 10**8)
    ... except ValueError as ve:
    ...     print(ve)
    Invalid fraction range 12.0, 0.0001.

    >>> try:
    ...     ExponentialScheduleBasedOnRange(OneMax(10), 0.9, 0.0001, 1)
    ... except ValueError as ve:
    ...     print(ve)
    n_steps=1 is invalid, must be in 2..1000000000000000.
    """

    def __init__(self, f: Objective, start_range_frac: float,
                 end_range_frac: float, n_steps: int,
                 min_range: int | float = _DEFAULT_MIN_RANGE,
                 max_range: int | float = _DEFAULT_MAX_RANGE) -> None:
        """
        Initialize the range-based exponential schedule.

        :param f: the objective function whose range we will use
        :param start_range_frac: the starting fraction of the range to use for
            the temperature
        :param end_range_frac: the end fraction of the range to use for the
            temperature
        :param n_steps: the number of steps until the end range should be
            reached
        """
        f = check_objective(f)
        if not isinstance(start_range_frac, float):
            raise type_error(start_range_frac, "start_range_frac", float)
        if not isinstance(end_range_frac, float):
            raise type_error(end_range_frac, "end_range_frac", float)
        if not (isfinite(start_range_frac) and isfinite(end_range_frac) and (
                1 >= start_range_frac > end_range_frac >= 0)):
            raise ValueError("Invalid fraction range "
                             f"{start_range_frac}, {end_range_frac}.")
        if not isinstance(max_range, int | float):
            raise type_error(max_range, "max_range", (int, float))
        if not isinstance(min_range, int | float):
            raise type_error(min_range, "min_range", (int, float))
        if not (0 < min_range < max_range):
            raise ValueError(
                f"Invalid range delimiters {min_range}, {max_range}.")
        #: the start objective range fraction
        self.start_range_frac: Final[float] = start_range_frac
        #: the end objective range fraction
        self.end_range_frac: Final[float] = end_range_frac
        #: the minimum objective range
        self.min_range: Final[int | float] = min_range
        #: the maximum objective range
        self.max_range: Final[int | float] = max_range
        #: the number of steps that we will perform until reaching the end
        #: range fraction temperature
        self.n_steps: Final[int] = check_int_range(
            n_steps, "n_steps", 2, 1_000_000_000_000_000)

        #: the name of the objective function used
        self.used_objective: Final[str] = enforce_non_empty_str_without_ws(
            str(f))

        flb: Final[float | int] = f.lower_bound()
        fub: Final[float | int] = f.upper_bound()
        f_range: float | int = 1

        if isfinite(flb):
            if isfinite(fub):
                if flb >= fub:
                    raise ValueError(
                        "objective function lower bound >= upper bound: "
                        f"{flb}, {fub}?")
                f_range = fub - flb
                if not isfinite(f_range) or (f_range <= 0):
                    raise ValueError(
                        f"Invalid bound range: {fub} - {flb} = {f_range}")
            else:
                f_range = max(abs(flb), 1)
        elif isfinite(fub):
            f_range = max(abs(fub), 1)
        f_range = min(max_range, max(min_range, f_range))

        #: the upper bound used for the objective range computation
        self.f_upper_bound: Final[int | float] = fub
        #: the lower bound used for the objective range computation
        self.f_lower_bound: Final[int | float] = flb
        #: The range of the objective function as used for the temperature
        #: computation.
        self.f_range: Final[int | float] = f_range

        #: the start temperature
        t0: Final[float] = start_range_frac * f_range
        te: Final[float] = end_range_frac * f_range
        if not (isfinite(t0) and isfinite(te) and (t0 > te)):
            raise ValueError(
                f"Invalid range {start_range_frac}, {end_range_frac}, "
                f"{f_range} leading to temperatures {t0}, {te}.")
        #: the end temperature
        self.te: Final[float] = te

        epsilon: Final[float] = 1 - (te / t0) ** (1 / (n_steps - 1))
        if not (isfinite(epsilon) and (0 < epsilon < 1) and (
                0 < (1 - epsilon) < 1)):
            raise ValueError(
                f"Invalid computed epsilon {epsilon} resulting from range "
                f"{start_range_frac}, {end_range_frac}, {f_range} leading "
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
        ...         ExponentialScheduleBasedOnRange(
        ...             OneMax(10), 0.1, 0.01, 10**8).log_parameters_to(kv)
        ...     text = l.get_log()
        >>> text[1]
        'name: expR0d1_0d01'
        >>> text[3]
        'T0: 1'
        >>> text[4]
        'e: 2.3025850892643973e-8'
        >>> len(text)
        21
        """
        super().log_parameters_to(logger)
        logger.key_value("startRangeFrac", self.start_range_frac)
        logger.key_value("endRangeFrac", self.end_range_frac)
        logger.key_value("maxRange", self.max_range)
        logger.key_value("minRange", self.min_range)
        logger.key_value("usedObjective", self.used_objective)
        logger.key_value("fLb", self.f_lower_bound)
        logger.key_value("fUb", self.f_upper_bound)
        logger.key_value("nSteps", self.n_steps)
        logger.key_value("fRange", self.f_range)
        logger.key_value("te", self.te)

    def __str__(self) -> str:
        """
        Get the string representation of the configured exponential schedule.

        :returns: the name of this schedule

        >>> from moptipy.examples.bitstrings.onemax import OneMax
        >>> ExponentialScheduleBasedOnRange(OneMax(10), 0.01, 0.0001, 10**8)
        expR0d01_0d0001
        """
        base: Final[str] = (
            f"expR{num_to_str_for_name(self.start_range_frac)}_"
            f"{num_to_str_for_name(self.end_range_frac)}")
        if (self.min_range != _DEFAULT_MIN_RANGE) or (
                self.max_range != _DEFAULT_MAX_RANGE):
            if self.min_range == _DEFAULT_MIN_RANGE:
                return f"{base}_{num_to_str_for_name(self.max_range)}"
            return (f"{base}_{num_to_str_for_name(self.min_range)}_"
                    f"{num_to_str_for_name(self.max_range)}")
        return base
