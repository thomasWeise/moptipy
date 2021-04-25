"""Some internal helper functions."""

from dataclasses import dataclass
from typing import Final, Union, Optional, Tuple

from math import inf
import moptipy.utils.logging as logging
from moptipy.utils.nputils import rand_seed_check

#: The key for the total number of runs.
KEY_N: Final[str] = "n"

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


def check_time_unit(time_unit: str) -> str:
    """
    Check that the time unit is OK.

    :param str time_unit: the time unit
    :return: the time unit string
    :rtype: str
    """
    if time_unit in (TIME_UNIT_FES, TIME_UNIT_MILLIS):
        return time_unit
    raise ValueError(
        f"Invalid time unit '{time_unit}', only {TIME_UNIT_FES} "
        f"and {TIME_UNIT_MILLIS} are permitted.")


def check_f_name(f_name: str) -> str:
    """
    Check whether an objective value name is valid.

    :param str f_name: the name of the objective function dimension
    :return: the name of the objective function dimension
    :rtype: str
    """
    if f_name in (F_NAME_RAW, F_NAME_SCALED, F_NAME_NORMALIZED):
        return f_name
    raise ValueError(
        f"Invalid f name '{f_name}', only {F_NAME_RAW}, "
        f"{F_NAME_SCALED}, and {F_NAME_NORMALIZED} are permitted.")


@dataclass(frozen=True, init=False, order=True)
class PerRunData:
    """An immutable record of information over a single run."""

    #: The algorithm that was applied.
    algorithm: str

    #: The problem instance that was solved.
    instance: str

    #: The seed of the random number generator.
    rand_seed: int

    def __init__(self,
                 algorithm: str,
                 instance: str,
                 rand_seed: int):
        """
        Create a per-run data record.

        :param str algorithm: the algorithm name
        :param str instance: the instance name
        :param int rand_seed: the random seed
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


@dataclass(frozen=True, init=False, order=True)
class MultiRunData:
    """
    A class that represents statistics over a set of runs.

    If one algorithm*instance is used, then `algorithm` and `instance` are
    defined. Otherwise, only the parameter which is the same over all recorded
    runs is defined.
    """

    #: The algorithm that was applied, if the same over all runs.
    algorithm: Optional[str]
    #: The problem instance that was solved, if the same over all runs.
    instance: Optional[str]
    #: The number of runs over which the statistic information is computed.
    n: int

    def __init__(self,
                 algorithm: Optional[str],
                 instance: Optional[str],
                 n: int):
        """
        Create the dataset of an experiment-setup combination.

        :param Optional[str] algorithm: the algorithm name, if all runs are
            with the same algorithm
        :param Optional[str] instance: the instance name, if all runs are
            on the same instance
        :param int n: the total number of runs
        """
        if algorithm is not None:
            if algorithm != logging.sanitize_name(algorithm):
                raise ValueError(f"Invalid algorithm '{algorithm}'.")
        object.__setattr__(self, "algorithm", algorithm)

        if instance is not None:
            if instance != logging.sanitize_name(instance):
                raise ValueError(f"Invalid instance '{instance}'.")
        object.__setattr__(self, "instance", instance)

        if not isinstance(n, int):
            raise TypeError(f"n must be int, but is {type(n)}.")
        if n <= 0:
            raise ValueError(f"n must be > 0, but is {n}.")
        object.__setattr__(self, "n", n)


@dataclass(frozen=True, init=False, order=True)
class MultiRun2DData(MultiRunData):
    """A multi-run data based on one time and one objective dimension."""

    #: The unit of the time axis.
    time_unit: str
    #: the name of the objective value axis.
    f_name: str

    def __init__(self,
                 algorithm: Optional[str],
                 instance: Optional[str],
                 n: int,
                 time_unit: str,
                 f_name: str):
        """
        Create multi-run data based on one time and one objective dimension.

        :param Optional[str] algorithm: the algorithm name, if all runs are
            with the same algorithm
        :param Optional[str] instance: the instance name, if all runs are
            on the same instance
        :param int n: the total number of runs
        :param str time_unit: the time unit
        :param str f_name: the objective dimension name
        """
        super().__init__(algorithm, instance, n)

        object.__setattr__(self, "time_unit", check_time_unit(time_unit))
        object.__setattr__(self, "f_name", check_f_name(f_name))


def get_instance(obj: Union[PerRunData, MultiRunData]) -> Optional[str]:
    """
    Get the instance of a given object.

    :param Union[PerRunData, MultiRunData] obj: the object
    :return: the instance string, or `None` if no instance is specified
    :rtype: Optional[str]
    """
    return obj.instance


def get_algorithm(obj: Union[PerRunData, MultiRunData]) -> Optional[str]:
    """
    Get the algorithm of a given object.

    :param Union[PerRunData, MultiRunData] obj: the object
    :return: the algorithm string, or `None` if no algorithm is specified
    :rtype: Optional[str]
    """
    return obj.algorithm


def sort_key(obj: Union[PerRunData, MultiRunData]) -> \
        Tuple[str, str, str, int, int, str, str, float]:
    """
    Get a default sort key for the given object.

    The sort key is a tuple with well-defined field elements that should
    allow for a default and consistent sorting over many different elements of
    the experiment evaluation data API. Sorting should work also for lists
    containing elements of different classes.

    :param Union[PerRunData, MultiRunData] obj: the object
    :return: the sort key
    :rtype: Tuple[str, str, int, int, str, str, float]
    """
    if hasattr(obj, "goal_f"):
        goal_f = getattr(obj, "goal_f")
    else:
        goal_f = None
    if goal_f is None:
        goal_f = inf

    return obj.__class__.__name__, \
        "" if obj.algorithm is None else obj.algorithm, \
        "" if obj.instance is None else obj.instance, \
        obj.n if isinstance(obj, MultiRunData) else 0, \
        obj.rand_seed if isinstance(obj, PerRunData) else 0, \
        obj.time_unit if isinstance(obj, MultiRun2DData) else "", \
        obj.f_name if isinstance(obj, MultiRun2DData) else "", \
        goal_f
