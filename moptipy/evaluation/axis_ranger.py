"""A utility to specify axis ranges."""
import sys
from math import inf, isfinite
from typing import Callable, Final

import numpy as np
from matplotlib.axes import Axes  # type: ignore

from moptipy.api.logging import (
    KEY_BEST_F,
    KEY_LAST_IMPROVEMENT_FE,
    KEY_LAST_IMPROVEMENT_TIME_MILLIS,
    KEY_TOTAL_FES,
    KEY_TOTAL_TIME_MILLIS,
)
from moptipy.evaluation.base import (
    F_NAME_NORMALIZED,
    F_NAME_RAW,
    F_NAME_SCALED,
    TIME_UNIT_FES,
    TIME_UNIT_MILLIS,
)
from moptipy.evaluation.end_statistics import KEY_ERT_FES, KEY_ERT_TIME_MILLIS
from moptipy.utils.types import type_error

#: The internal minimum float value for log-scaled axes.
_MIN_LOG_FLOAT: Final[float] = sys.float_info.min


class AxisRanger:
    """An object for simplifying axis range computations."""

    def __init__(self,
                 chosen_min: float | None = None,
                 chosen_max: float | None = None,
                 use_data_min: bool = True,
                 use_data_max: bool = True,
                 log_scale: bool = False,
                 log_base: float | None = None):
        """
        Initialize the axis ranger.

        :param chosen_min: the chosen minimum
        :param chosen_max: the chosen maximum
        :param use_data_min: use the minimum found in the data?
        :param use_data_max:  use the maximum found in the data?
        :param log_scale: should the axis be log-scaled?
        :param log_base: the base to be used for the logarithm
        """
        if not isinstance(log_scale, bool):
            raise type_error(log_scale, "log_scale", bool)
        #: Should the axis be log-scaled?
        self.log_scale: Final[bool] = log_scale

        self.__log_base: Final[float | None] = \
            log_base if self.log_scale else None
        if self.__log_base is not None:
            if not isinstance(log_base, float):
                raise type_error(log_base, "log_base", float)
            if log_base <= 1.0:
                raise ValueError(f"log_base must be > 1, but is {log_base}.")

        if chosen_min is not None:
            if not isinstance(chosen_min, float | int):
                raise type_error(chosen_min, "chosen_min", (int, float))
            chosen_min = float(chosen_min)
            if not isfinite(chosen_min):
                raise ValueError(f"chosen_min cannot be {chosen_min}.")
            if self.log_scale and (chosen_min <= 0):
                raise ValueError(
                    f"if log_scale={self.log_scale}, then chosen_min must "
                    f"be > 0, but is {chosen_min}.")

        #: The pre-defined, chosen minimum axis value.
        self.__chosen_min: Final[float | None] = chosen_min

        if chosen_max is not None:
            if not isinstance(chosen_max, float | int):
                raise type_error(chosen_max, "chosen_max", (int, float))
            chosen_max = float(chosen_max)
            if not isfinite(chosen_max):
                raise ValueError(f"chosen_max cannot be {chosen_max}.")
            if (self.__chosen_min is not None) and \
                    (chosen_max <= self.__chosen_min):
                raise ValueError(f"If chosen_min is {self.__chosen_min}, then"
                                 f" chosen_max cannot be {chosen_max}.")

        #: The pre-defined, chosen maximum axis value.
        self.__chosen_max: Final[float | None] = chosen_max

        if not isinstance(use_data_min, bool):
            raise type_error(use_data_min, "use_data_min", bool)
        #: Should we use the data min value?
        self.__use_data_min: Final[bool] = use_data_min

        if not isinstance(use_data_max, bool):
            raise type_error(use_data_max, "use_data_max", bool)
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

    def register_array(self, data: np.ndarray) -> None:
        """
        Register a data array.

        :param data: the data to register
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

        :param value: the data to register
        """
        if isfinite(value):
            if self.__use_data_min and (
                    (value < self.__detected_min)
                    and ((value > 0.0) or (not self.log_scale))):
                self.__detected_min = value
                self.__has_detected_min = True
            if self.__use_data_max and (value > self.__detected_max):
                self.__detected_max = value
                self.__has_detected_max = True

    def pad_detected_range(self, pad_min: bool = False,
                           pad_max: bool = False) -> None:
        """
        Add some padding to the current detected range.

        This function increases the current detected or chosen maximum value
        and/or decreases the current detected minimum by a small amount. This
        can be useful when we want to plot stuff that otherwise would become
        invisible because it would be directly located at the boundary of a
        plot.

        This function works by computing a slightly smaller/larger value than
        the current detected minimum/maximum and then passing it to
        :meth:`register_value`. It can only work if the end(s) chosen for
        padding are in "detect" mode and the other end is either in "detect"
        or "chosen" mode.

        This method should be called *only* once and *only* after all data has
        been registered (via :meth:`register_value` :meth:`register_array`)
        and before calling :meth:`apply`.

        :param pad_min: should we pad the minimum?
        :param pad_max: should we pad the maximum?

        :raises ValueError: if this axis ranger is not configured to use a
            detected minimum/maximum or does not have a detected
            minimum/maximum or any other invalid situation occurs
        """
        if not isinstance(pad_min, bool):
            raise type_error(pad_min, "pad_min", bool)
        if not isinstance(pad_max, bool):
            raise type_error(pad_max, "pad_max", bool)
        if not (pad_min or pad_max):
            return

        max_value: float
        min_value: float
        if self.__use_data_min:
            if not self.__has_detected_min:
                raise ValueError("No minimum detected so far.")
            min_value = self.__detected_min
        else:
            if pad_min:
                raise ValueError("Can only pad minimum if use_data_min.")
            if self.__chosen_min is None:
                raise ValueError("Chosen min is None!")
            min_value = self.__chosen_min

        if self.__use_data_max:
            if not self.__has_detected_max:
                raise ValueError("No maximum detected so far.")
            max_value = self.__detected_max
        else:
            if pad_max:
                raise ValueError("Can only pad maximum if use_data_max.")
            if self.__chosen_max is None:
                raise ValueError("Chosen max is None!")
            max_value = self.__chosen_max

        if min_value >= max_value:
            raise ValueError(
                f"minimum={min_value} while maximum={max_value}.")

        new_max: float
        if pad_max:
            if max_value >= inf:
                return
            new_max = max_value + (3.0 * (max_value - min_value)) / 100.0
            if not isfinite(new_max) or (new_max <= max_value):
                raise ValueError(f"invalid padded max={new_max} at min="
                                 f"{min_value} and max={max_value}.")
            self.register_value(new_max)
        else:
            new_max = max_value

        new_min: float
        if pad_min:
            if min_value <= -inf:
                return
            new_min = min_value - (3.0 * (max_value - min_value)) / 100.0
            if self.log_scale and (new_min <= 0.0 < min_value):
                new_min = 0.5 * min_value
            if not isfinite(new_min) or (new_min >= min_value):
                raise ValueError(f"invalid padded min={new_min} at min="
                                 f"{min_value} and max={max_value}.")
            self.register_value(new_min)
        else:
            new_min = min_value

        if new_min > new_max:
            raise ValueError(f"new_min={new_min}, new_max={new_max}??")

    def apply(self, axes: Axes, which_axis: str) -> None:
        """
        Apply this axis ranger to the given axis.

        :param axes: the axes object to which the ranger shall be applied
        :param which_axis: the axis to which it should be applied, either
            `"x"` or `"y"` or both (`"xy"`)
        """
        if not isinstance(which_axis, str):
            raise type_error(which_axis, "which_axis", str)

        for is_x_axis in (True, False):
            if ("x" if is_x_axis else "y") not in which_axis:
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
                if not self.__has_detected_min:
                    raise ValueError("No minimum in data detected.")
                use_min = self.__detected_min
                replace_range = True

            if self.__chosen_max is not None:
                use_max = self.__chosen_max
                replace_range = True
            elif self.__use_data_max:
                if not self.__has_detected_max:
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
                        axes.semilogx(base=self.__log_base)
                elif self.__log_base is None:
                    axes.semilogy()
                else:
                    axes.semilogy(base=self.__log_base)

    def get_pinf_replacement(self) -> float:
        """
        Get a reasonable finite value that can replace positive infinity.

        :return: a reasonable finite value that can be used to replace
            positive infinity
        """
        data_max: float = 0.0
        if self.__chosen_max is not None:
            data_max = self.__chosen_max
        elif self.__has_detected_max:
            data_max = self.__detected_max
        return min(1e100, max(1e70, 1e5 * data_max))

    def get_0_replacement(self) -> float:
        """
        Get a reasonable positive finite value that can replace `0`.

        :return: a reasonable finite value that can be used to replace
            `0`
        """
        data_min: float = 1e-100
        if self.__chosen_min is not None:
            data_min = self.__chosen_min
        elif self.__has_detected_min:
            data_min = self.__detected_min
        return max(1e-100, min(1e-70, 1e-5 * data_min))

    @staticmethod
    def for_axis(name: str,
                 chosen_min: float | None = None,
                 chosen_max: float | None = None,
                 use_data_min: bool | None = None,
                 use_data_max: bool | None = None,
                 log_scale: bool | None = None,
                 log_base: float | None = None) -> "AxisRanger":
        """
        Create a default axis ranger based on the axis type.

        The axis ranger will use the minimal values and log scaling options
        that usually make sense for the dimension, unless overridden by the
        optional arguments.

        :param name: the axis type name, supporting `"ms"`, `"FEs"`,
            `"plainF"`, `"scaledF"`, and `"normalizedF"`
        :param chosen_min: the chosen minimum
        :param chosen_max: the chosen maximum
        :param use_data_min: should the data minimum be used
        :param use_data_max: should the data maximum be used
        :param log_scale: the log scale indicator
        :param log_base: the log base
        :return: the `AxisRanger`
        """
        if not isinstance(name, str):
            raise type_error(name, "axis name", str)

        __log: bool = False
        __min: float | None = None
        __max: float | None = None
        __data_min: bool = chosen_min is None
        __data_max: bool = chosen_max is None

        if name in (TIME_UNIT_MILLIS, KEY_LAST_IMPROVEMENT_TIME_MILLIS,
                    KEY_TOTAL_TIME_MILLIS, KEY_ERT_TIME_MILLIS):
            if chosen_min is not None:
                if (chosen_min < 0) or (not isfinite(chosen_min)):
                    raise ValueError("chosen_min must be >= 0 for axis "
                                     f"type {name}, but is {chosen_min}.")
                __log = (chosen_min > 0)
                if log_scale is not None:
                    if log_scale and (not __log):
                        raise ValueError(f"Cannot set log_scale={log_scale} "
                                         f"and chosen_min={chosen_min} for "
                                         f"axis type {name}.")
                    __log = log_scale
            elif log_scale is None:
                __log = True
            else:
                __log = log_scale

            __min = (1 if __log else 0) if chosen_min is None else chosen_min

            if use_data_max is not None:
                __data_max = use_data_max

            __data_min = False if use_data_min is None else use_data_min

            if chosen_max is not None:
                __max = chosen_max

            return AxisRanger(__min, __max, __data_min, __data_max,
                              __log, log_base if __log else None)

        if name in (TIME_UNIT_FES, KEY_LAST_IMPROVEMENT_FE, KEY_TOTAL_FES,
                    KEY_ERT_FES):
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

            __data_min = False if use_data_min is None else use_data_min

            return AxisRanger(__min, chosen_max, __data_min, __data_max,
                              __log, log_base if __log else None)

        if name in (F_NAME_RAW, KEY_BEST_F):
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
            raise ValueError(f"Axis type {name!r} is unknown.")

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
    def for_axis_func(chosen_min: float | None = None,
                      chosen_max: float | None = None,
                      use_data_min: bool | None = None,
                      use_data_max: bool | None = None,
                      log_scale: bool | None = None,
                      log_base: float | None = None) -> Callable:
        """
        Generate a function that provides the default per-axis ranger.

        :param chosen_min: the chosen minimum
        :param chosen_max: the chosen maximum
        :param use_data_min: should the data minimum be used
        :param use_data_max: should the data maximum be used
        :param log_scale: the log scale indicator
        :param log_base: the log base
        :return: a function in the shape of :meth:`for_axis` with the
            provided defaults
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
