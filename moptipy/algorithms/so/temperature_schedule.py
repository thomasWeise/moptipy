"""A temperature schedule as needed by simulated annealing."""

from math import e, isfinite, log
from typing import Final

from moptipy.api.component import Component
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.strings import num_to_str_for_name
from moptipy.utils.types import type_error


# start schedule
class TemperatureSchedule(Component):
    """The base class for temperature schedules."""

    def __init__(self, start_temperature: float) -> None:
        # end schedule
        """
        Initialize the temperature schedule.

        :param start_temperature: the starting temperature
        """
        super().__init__()
        if not isinstance(start_temperature, float):
            raise type_error(start_temperature, "start_temperature", float)
        if (not isfinite(start_temperature)) or (start_temperature < 0.0):
            raise ValueError(
                f"start_temperature cannot be {start_temperature}.")
# start schedule
        #: the starting temperature
        self.start_temperature: Final[float] = start_temperature

    def temperature(self, tau: int) -> float:
        """
        Compute the temperature at iteration `tau`.

        :param tau: the iteration index, starting with 0 at the first
            comparison of two solutions, at which point the starting
            temperature should be returned
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
        logger.key_value("T0", self.start_temperature)


# start exponential
class ExponentialSchedule(TemperatureSchedule):
    """The exponential temperature schedule."""

    def __init__(self, start_temperature: float, epsilon: float) -> None:
        """
        Initialize the exponential temperature schedule.

        :param start_temperature: the starting temperature
        :param epsilon: the epsilon parameter of the schedule
        """
        super().__init__(start_temperature)
# end exponential
        if not isinstance(epsilon, float):
            raise type_error(epsilon, "epsilon", float)
        if (not isfinite(epsilon)) or (not (0.0 < epsilon < 1.0)):
            raise ValueError(
                f"epsilon cannot be {epsilon}, must be in (0, 1).")
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

        :param tau: the iteration index, starting with 0 at the first
            comparison of two solutions, at which point the starting
            temperature should be returned
        :returns: the temperature

        >>> s = ExponentialSchedule(100.0, 0.5)
        >>> s.temperature(0)
        100.0
        >>> s.temperature(1)
        50.0
        >>> s.temperature(10)
        0.09765625
        """
        return self.start_temperature * (self.__one_minus_epsilon ** tau)
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
        return f"exp{num_to_str_for_name(self.start_temperature)}_" \
               f"{num_to_str_for_name(self.epsilon)}"


# start logarithmic
class LogarithmicSchedule(TemperatureSchedule):
    """The logarithmic temperature schedule."""

    def __init__(self, start_temperature: float, epsilon: float) -> None:
        """
        Initialize the logarithmic temperature schedule.

        :param start_temperature: the starting temperature
        :param epsilon: the epsilon parameter of the schedule
        """
        super().__init__(start_temperature)
# end logarithmic
        if not isinstance(epsilon, float):
            raise type_error(epsilon, "epsilon", float)
        if (not isfinite(epsilon)) or (not (0.0 < epsilon < 1.0)):
            raise ValueError(
                f"epsilon cannot be {epsilon}, must be in (0, 1).")
# start logarithmic
        #: the epsilon parameter of the logarithmic schedule
        self.epsilon: Final[float] = epsilon

    def temperature(self, tau: int) -> float:
        """
        Compute the temperature at iteration `tau`.

        :param tau: the iteration index, starting with 0 at the first
            comparison of two solutions, at which point the starting
            temperature should be returned
        :returns: the temperature

        >>> s = LogarithmicSchedule(100.0, 0.5)
        >>> s.temperature(0)
        100.0
        >>> s.temperature(1)
        85.55435113150568
        >>> s.temperature(10)
        48.93345190925178
        """
        return self.start_temperature / log(e + (tau * self.epsilon))
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
        return f"ln{num_to_str_for_name(self.start_temperature)}_" \
               f"{num_to_str_for_name(self.epsilon)}"
