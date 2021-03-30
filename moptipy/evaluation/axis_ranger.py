"""A utility to specify axis ranges."""
import sys
from math import isfinite, inf
from typing import Optional, Final, Iterable

import numpy as np
from matplotlib.axes import Axes  # type: ignore

#: The internal minimum float value for log-scaled axes.
_MIN_LOG_FLOAT: Final[float] = sys.float_info.min


class AxisRanger:
    """An object for simplifying axis range computations."""

    def __init__(self,
                 chosen_min: Optional[float] = None,
                 chosen_max: Optional[float] = None,
                 use_data_min: bool = True,
                 use_data_max: bool = True,
                 log_scale: bool = False,
                 log_base: Optional[float] = None):
        """
        Initialize the axis ranger.

        :param Optional[float] chosen_min: the chosen minimum
        :param Optional[float] chosen_max: the chosen maximum
        :param bool use_data_min: use the minimum found in the data?
        :param bool use_data_max:  use the maximum found in the data?
        :param bool log_scale: should the axis be log-scaled?
        :param float log_base: the base to be used for the logarithm
        """
        if not isinstance(log_scale, bool):
            raise TypeError(
                f"log_scale must be bool, but is {type(log_scale)}.")
        #: Should the axis be log-scaled?
        self.__log_scale: Final[bool] = log_scale

        self.__log_base: Final[Optional[float]] = \
            log_base if self.__log_scale else None
        if self.__log_base is not None:
            if not isinstance(log_base, float):
                raise TypeError("log_base must be float if specified, "
                                f"but encountered {type(log_base)}.")
            if log_base <= 1.0:
                raise ValueError(f"log_base must be > 1, but is {log_base}.")

        if chosen_min is not None:
            if not isinstance(chosen_min, (float, int)):
                raise TypeError("chosen_min must be float, int, or None, "
                                f"but is {type(chosen_min)}.")
            chosen_min = float(chosen_min)
            if not isfinite(chosen_min):
                raise ValueError(f"chosen_min cannot be {chosen_min}.")
            if self.__log_scale and (chosen_min <= 0):
                raise ValueError(
                    f"if log_scale={self.__log_scale}, then chosen_min must "
                    f"be > 0, but is {chosen_min}.")

        #: The pre-defined, chosen minimum axis value.
        self.__chosen_min: Final[Optional[float]] = chosen_min

        if chosen_max is not None:
            if not isinstance(chosen_max, (float, int)):
                raise TypeError("chosen_max must be float, int, or None, "
                                f"but is {type(chosen_max)}.")
            chosen_max = float(chosen_max)
            if not isfinite(chosen_max):
                raise ValueError(f"chosen_max cannot be {chosen_max}.")
            if self.__chosen_min is not None:
                if chosen_max <= self.__chosen_min:
                    raise ValueError(
                        f"If chosen_min is {self.__chosen_min}, then "
                        f"chosen_max cannot be {chosen_max}.")

        #: The pre-defined, chosen maximum axis value.
        self.__chosen_max: Final[Optional[float]] = chosen_max

        if not isinstance(use_data_min, bool):
            raise TypeError(
                f"use_data_min must be bool, but is {type(use_data_min)}.")
        #: Should we use the data min value?
        self.__use_data_min: Final[bool] = use_data_min

        if not isinstance(use_data_max, bool):
            raise TypeError(
                f"use_data_max must be bool, but is {type(use_data_max)}.")
        #: Should we use the data max value?
        self.__use_data_max: Final[bool] = use_data_max

        #: The minimum detected from the data.
        self.__detected_min: float = inf

        #: The maximum detected from the data.
        self.__detected_max: float = _MIN_LOG_FLOAT if self.__log_scale \
            else -inf

        #: Did we detect a minimum?
        self.__has_detected_min = False
        #: Did we detect a maximum?
        self.__has_detected_max = False

    def reset(self) -> None:
        """Reset the detected data, making the object ready for reuse."""
        self.__has_detected_min = False
        self.__has_detected_max = False
        self.__detected_min = inf
        self.__detected_max = _MIN_LOG_FLOAT if self.__log_scale else -inf

    def register_array(self, data: np.ndarray) -> None:
        """
        Register a data array.

        :param np.ndarray data: the data to register
        """
        if self.__use_data_min:
            self.register_value(float(data.min()))
        if self.__use_data_max:
            self.register_value(float(data.max()))

    def register_value(self, value: float) -> None:
        """
        Register a single value.

        :param float value: the data to register
        """
        if self.__use_data_min:
            if (value < self.__detected_min) and \
                    ((value > 0.0) or (not self.__log_scale)):
                self.__detected_min = value
                self.__has_detected_min = True
        if self.__use_data_max:
            if value > self.__detected_max:
                self.__detected_max = value
                self.__has_detected_max = True

    def register_seq(self, seq: Iterable[float]) -> None:
        """
        Register a sequence of values.

        :param Iterable[float] seq: the data to register
        """
        for value in seq:
            self.register_value(value)

    def apply(self, axes: Axes,
              which_axis: str) -> None:
        """
        Apply this axis ranger to the given axis.

        :param Axes axes: the axes object to apply the ranger to
        :param str which_axis: the axis to which it should be applied, either
            "x" or "y" or both
        """
        if not isinstance(which_axis, str):
            raise TypeError(
                f"which_axis must be str but is {type(which_axis)}.")

        for is_x_axis in [True, False]:
            if not (("x" if is_x_axis else "y") in which_axis):
                continue

            use_min, use_max = \
                axes.get_xlim() if is_x_axis else axes.get_ylim()

            if not isfinite(use_min):
                raise ValueError(f"Minimum data interval cannot be {use_min}.")
            if not isfinite(use_max):
                raise ValueError(f"Maximum data interval cannot be {use_max}.")
            if use_max <= use_min:
                raise ValueError(f"Invalid axis range[{use_min},{use_max}].")

            replace_range = False

            if self.__chosen_min is not None:
                use_min = self.__chosen_min
                replace_range = True
            elif self.__use_data_min:
                if not self.__detected_min:
                    raise ValueError("No minimum in data detected.")
                use_min = self.__detected_min
                replace_range = True

            if self.__chosen_max is not None:
                use_max = self.__chosen_max
                replace_range = True
            elif self.__use_data_max:
                if not self.__detected_max:
                    raise ValueError("No maximum in data detected.")
                use_max = self.__detected_max
                replace_range = True

            if replace_range:
                if use_min >= use_max:
                    raise ValueError(
                        f"Invalid computed range [{use_min},{use_max}].")
                if is_x_axis:
                    axes.set_xlim(use_min, use_max)
                else:
                    axes.set_ylim(use_min, use_max)

            if self.__log_scale:
                if use_min <= 0:
                    raise ValueError("minimum must be positive if log scale "
                                     f"is defined, but found {use_min}.")
                if is_x_axis:
                    if self.__log_base is None:
                        axes.semilogx()
                    else:
                        axes.semilogx(self.__log_base)
                else:
                    if self.__log_base is None:
                        axes.semilogy()
                    else:
                        axes.semilogy(self.__log_base)
