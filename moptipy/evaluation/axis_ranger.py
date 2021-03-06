"""A utility to specify axis ranges."""
import sys
from math import isfinite, inf
from typing import Optional, Final, Iterable, Callable

import numpy as np
from matplotlib.axes import Axes  # type: ignore

from moptipy.evaluation.base import TIME_UNIT_MILLIS, TIME_UNIT_FES, \
    F_NAME_RAW, F_NAME_SCALED, F_NAME_NORMALIZED

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
        self.log_scale: Final[bool] = log_scale

        self.__log_base: Final[Optional[float]] = \
            log_base if self.log_scale else None
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
            if self.log_scale and (chosen_min <= 0):
                raise ValueError(
                    f"if log_scale={self.log_scale}, then chosen_min must "
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
        self.__detected_max: float = _MIN_LOG_FLOAT if self.log_scale \
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
        self.__detected_max = _MIN_LOG_FLOAT if self.log_scale else -inf

    def register_array(self, data: np.ndarray) -> None:
        """
        Register a data array.

        :param np.ndarray data: the data to register
        """
        if self.__use_data_min or self.__use_data_max:
            d = data[np.isfinite(data)]
            if self.__use_data_min:
                self.register_value(float(d.min()))
            if self.__use_data_max:
                self.register_value(float(d.max()))

    def register_value(self, value: float) -> None:
        """
        Register a single value.

        :param float value: the data to register
        """
        if isfinite(value):
            if self.__use_data_min:
                if (value < self.__detected_min) and \
                        ((value > 0.0) or (not self.log_scale)):
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

            if self.log_scale:
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

    def get_pinf_replacement(self) -> float:
        """
        Get a reasonable finite value that can replace positive infinity.

        :return: a reasonable finite value that can be used to replace
            positive infinity
        :rtype: float
        """
        data_max: float = 0.0
        if self.__chosen_max is not None:
            data_max = self.__chosen_max
        elif self.__detected_max is not None:
            data_max = self.__detected_max
        return min(1e100, max(1e70, 1e5 * data_max))

    def get_0_replacement(self) -> float:
        """
        Get a reasonable positive finite value that can replace `0`.

        :return: a reasonable finite value that can be used to replace
            `0`
        :rtype: float
        """
        data_min: float = 1e-100
        if self.__chosen_min is not None:
            data_min = self.__chosen_min
        elif self.__detected_min is not None:
            data_min = self.__detected_min
        return max(1e-100, min(1e-70, 1e-5 * data_min))

    @staticmethod
    def for_axis(name: str,
                 chosen_min: Optional[float] = None,
                 chosen_max: Optional[float] = None,
                 use_data_min: Optional[bool] = None,
                 use_data_max: Optional[bool] = None,
                 log_scale: Optional[bool] = None,
                 log_base: Optional[float] = None) -> 'AxisRanger':
        """
        Create a default axis ranger based on the axis type.

        The axis ranger will use the minimal values and log scaling options
        that usually make sense for the dimension, unless overridden by the
        optional arguments.

        :param str name: the axis type name, supporting `"ms"`, `"FEs"`,
            `"plain"`, `"scaled"`, and `"normalized"`
        :param Optional[float] chosen_min: the chosen minimum
        :param Optional[float] chosen_max: the chosen maximum
        :param Optional[bool] use_data_min: should the data minimum be used
        :param Optional[bool] use_data_max: should the data maximum be used
        :param Optional[bool] log_scale: the log scale indicator
        :param Optional[float] log_base: the log base
        :return: the `AxisRanger`
        :rtype: AxisRanger
        """
        if not isinstance(name, str):
            raise TypeError(f"Axis name must be str, but is {type(name)}.")

        __log: bool = False
        __min: Optional[float] = None
        __max: Optional[float] = None
        __data_min: bool = chosen_min is None
        __data_max: bool = chosen_max is None

        if name == TIME_UNIT_MILLIS:
            if chosen_min is not None:
                if (chosen_min < 0) or (not isfinite(chosen_min)):
                    raise ValueError("chosen_min must be >= 0 for axis "
                                     f"type {name}, but is {chosen_min}.")
                __log = (chosen_min > 0)
                if log_scale is not None:
                    if log_scale and (not __log):
                        raise TypeError(f"Cannot set log_scale={log_scale} "
                                        f"and chosen_min={chosen_min} for "
                                        f"axis type {name}.")
                    __log = log_scale
            elif log_scale is None:
                __log = True
            else:
                __log = log_scale

            if chosen_min is None:
                __min = 1 if __log else 0
            else:
                __min = chosen_min

            if use_data_max is not None:
                __data_max = use_data_max
            if use_data_min is not None:
                __data_min = use_data_min
            else:
                __data_min = False

            if chosen_max is not None:
                __max = chosen_max

            return AxisRanger(__min, __max, __data_min, __data_max,
                              __log, log_base if __log else None)

        if name == TIME_UNIT_FES:
            if chosen_min is None:
                __min = 1
            else:
                if (chosen_min < 1) or (not isfinite(chosen_min)):
                    raise ValueError("chosen_min must be >= 1 for axis "
                                     f"type {name}, but is {chosen_min}.")
                __min = chosen_min
            __log = True if (log_scale is None) else log_scale

            if use_data_max is not None:
                __data_max = use_data_max
            if use_data_min is not None:
                __data_min = use_data_min
            else:
                __data_min = False

            return AxisRanger(__min, chosen_max, __data_min, __data_max,
                              __log, log_base if __log else None)

        if name == F_NAME_RAW:
            if use_data_max is not None:
                __data_max = use_data_max
            if use_data_min is not None:
                __data_min = use_data_min
            if log_scale is not None:
                __log = log_scale
            return AxisRanger(chosen_min, chosen_max, __data_min, __data_max,
                              __log, log_base if __log else None)

        if name == F_NAME_SCALED:
            __min = 1
        elif name == F_NAME_NORMALIZED:
            if (log_scale is None) or (not log_scale):
                __min = 0
        elif name == "ecdf":
            if (log_scale is None) or (not log_scale):
                __min = 0
            __max = 1
        else:
            raise ValueError(f"Axis type '{name}' is unknown.")

        if chosen_min is not None:
            __min = chosen_min

        if log_scale is not None:
            __log = log_scale
        if use_data_max is not None:
            __data_max = use_data_max
        if use_data_min is not None:
            __data_min = use_data_min

        return AxisRanger(__min, chosen_max, __data_min, __data_max,
                          __log, log_base if __log else None)

    @staticmethod
    def for_axis_func(chosen_min: Optional[float] = None,
                      chosen_max: Optional[float] = None,
                      use_data_min: Optional[bool] = None,
                      use_data_max: Optional[bool] = None,
                      log_scale: Optional[bool] = None,
                      log_base: Optional[float] = None) -> Callable:
        """
        Generate a function that provides the default per-axis ranger.

        :param Optional[float] chosen_min: the chosen minimum
        :param Optional[float] chosen_max: the chosen maximum
        :param Optional[bool] use_data_min: should the data minimum be used
        :param Optional[bool] use_data_max: should the data maximum be used
        :param Optional[bool] log_scale: the log scale indicator
        :param Optional[float] log_base: the log base
        :return: the a function in the shape of :meth:`for_axis` with the
            provided defaults
        :rtype: Callable
        """
        def __func(name: str,
                   cmi=chosen_min,
                   cma=chosen_max,
                   udmi=use_data_min,
                   udma=use_data_max,
                   ls=log_scale,
                   lb=log_base) -> AxisRanger:
            return AxisRanger.for_axis(name, cmi, cma, udmi, udma, ls, lb)

        return __func
