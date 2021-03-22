"""Progress data over a run."""
from dataclasses import dataclass
from typing import Final

import numpy as np

from moptipy.utils import logging
from moptipy.utils.nputils import rand_seed_check, is_np_int, is_np_float

#: The unit of the time axis if time is measured in milliseconds.
TIME_UNIT_MILLIS: Final[str] = "ms"
#: The unit of the time axis of time is measured in FEs
TIME_UNIT_FES: Final[str] = "FEs"

#: The name of the raw objective values data.
F_NAME_RAW: Final[str] = "plain"
#: The name of the scaled objective values data.
F_NAME_SCALED: Final[str] = "scaled"
#: The name of the normalized objective values data.
F_NAME_NORMALIZED: Final[str] = "normalized"


@dataclass(frozen=True, init=False, order=True)
class Progress:
    """An immutable record of progress information over a single run."""

    #: The algorithm that was applied.
    algorithm: str

    #: The problem instance that was solved.
    instance: str

    #: The seed of the random number generator.
    rand_seed: int

    #: The time axis data.
    time: np.ndarray

    #: The unit of the time axis.
    time_unit: str

    #: The objective value data.
    f: np.ndarray

    #: the name of the objective value axis.
    f_name: str

    def __init__(self,
                 algorithm: str,
                 instance: str,
                 rand_seed: int,
                 time: np.ndarray,
                 time_unit: str,
                 f: np.ndarray,
                 f_name: str):
        """
        Create a consistent instance of :class:`EndResult`.

        :param str algorithm: the algorithm name
        :param str instance: the instance name
        :param int rand_seed: the random seed
        :param np.array time: the time axis data
        :param str time_unit: the unit of the time axis
        :param np.array f: the objective value axis data
        :param str f_name: the name of the objective value axis data
        """
        if not isinstance(algorithm, str):
            raise TypeError(
                f"algorithm must be str, but is {type(algorithm)}.")
        if algorithm != logging.sanitize_name(algorithm):
            raise ValueError("In valid algorithm must name '{algorithm}'.")
        object.__setattr__(self, "algorithm", algorithm)

        if not isinstance(instance, str):
            raise TypeError(
                f"instance must be str, but is {type(instance)}.")
        if instance != logging.sanitize_name(instance):
            raise ValueError("In valid instance must name '{instance}'.")
        object.__setattr__(self, "instance", instance)
        object.__setattr__(self, "rand_seed", rand_seed_check(rand_seed))

        if not isinstance(time, np.ndarray):
            raise TypeError(
                f"time data must be np.array, but is {type(time)}.")
        if not is_np_int(time.dtype):
            raise TypeError("time data must be integer-valued.")
        tl = len(time)
        if tl <= 0:
            raise ValueError("time data must not be empty.")
        if tl > 1:
            if np.any(time[1:] <= time[:-1]):
                raise ValueError("time data must be strictly increasing.")

        object.__setattr__(self, "time", time)

        if not (time_unit in (TIME_UNIT_FES, TIME_UNIT_MILLIS)):
            raise ValueError(
                f"Invalid time unit '{time_unit}', only {TIME_UNIT_FES} "
                f"and {TIME_UNIT_MILLIS} are permitted.")
        object.__setattr__(self, "time_unit", time_unit)

        mintime = 1 if time_unit == TIME_UNIT_FES else 0
        if any(time < mintime):
            raise ValueError(f"No time value can be less than {mintime} if"
                             f" time unit is {time_unit}.")

        if not isinstance(f, np.ndarray):
            raise TypeError(
                f"f data must be np.array, but is {type(f)}.")
        if not (is_np_int(f.dtype) or is_np_float(f.dtype)):
            raise TypeError("f data must be integer- or float valued.")
        fl = len(f)
        if fl <= 0:
            raise ValueError("f data must not be empty.")
        if fl != tl:
            raise ValueError(f"Length {fl} of f data and length {tl} of "
                             "time data must be the same.")
        if fl > 1:
            if np.any(f[1:-1] >= f[:-2]):
                raise ValueError(
                    "f data must be strictly decreasing, with "
                    "only the entry being permitted as exception.")
            if f[-1] > f[-2]:
                raise ValueError(f"last f-value ({f[-1]}) cannot be greater"
                                 f"than second-to-last ({f[-2]}).")
        object.__setattr__(self, "f", f)

        if not (f_name in (F_NAME_RAW, F_NAME_SCALED, F_NAME_NORMALIZED)):
            raise ValueError(
                f"Invalid time unit '{time_unit}', only {TIME_UNIT_FES} "
                f"and {TIME_UNIT_MILLIS} are permitted.")
        object.__setattr__(self, "time_unit", time_unit)
