"""Some internal helper functions."""

from dataclasses import dataclass
from typing import Final
from typing import Optional

import moptipy.utils.logging as logging
from moptipy.utils.nputils import rand_seed_check

#: The key for the total number of runs.
KEY_N: Final[str] = "n"


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
        Create the end statistics of an experiment-setup combination.

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
