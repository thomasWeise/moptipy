"""A temperature schedule as needed by simulated annealing."""

from math import isfinite
from typing import Final

from moptipy.api.component import Component
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.types import type_error


# start schedule
class TemperatureSchedule(Component):
    """The base class for temperature schedules."""

    def __init__(self, start_temperature: float) -> None:
        """
        Initialize the temperature schedule.

        :param start_temperature: the starting temperature
        """
# end schedule
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

        :param tau: the iteration index
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
        """
        super().__init__(start_temperature)
# end exponential
        if not isinstance(epsilon, float):
            raise type_error(epsilon, "epsilon", float)
        if (not isfinite(epsilon)) or (not (0.0 < epsilon < 1.0)):
            raise ValueError(
                f"epsilon cannot be {epsilon}, must be in (0, 1).")
# start exponential
        #: store the epsilon parameter of the exponential schedule
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

        :param tau: the iteration index
        :returns: the temperature
        """
        return self.start_temperature * (self.__one_minus_epsilon ** (tau - 1))
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
        'name: ExponentialSchedule'
        >>> text[3]
        'T0: 0.2'
        >>> text[5]
        'e: 0.6'
        >>> len(text)
        8
        """
        super().log_parameters_to(logger)
        logger.key_value("e", self.epsilon)
