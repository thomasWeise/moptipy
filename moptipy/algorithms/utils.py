"""Some internal utilities for optimization."""
from typing import Final, Union


# start record
class Record:
    """
    A point in the search space, its quality, and creation time.

    A record stores a point in the search space :attr:`x` together with
    the corresponding objective value :attr:`f`. It also stores a
    "iteration index" :attr:`it`, i.e., the time when the point was
    created or modified.

    This allows for representing and storing solutions in a population.
    If the population is sorted, then individual records with better
    objective value will be moved to the beginning of the list. Ties are
    broken such that younger individuals (with higher :attr:`it` value)
    are preferred.
    """

    def __init__(self, x, f: Union[int, float]):
        """
        Create the record.

        :param x: the data structure for a point in the search space
        :param f: the corresponding objective value
        """
        #: the point in the search space
        self.x: Final = x
        #: the objective value corresponding to x
        self.f: Union[int, float] = f
        #: the iteration index when the record was created
        self.it: int = 0

    def __lt__(self, other):
        """Precedence if 1) better or b) equally good but younger."""
        f1: Final[Union[int, float]] = self.f
        f2: Final[Union[int, float]] = other.f
        return (f1 < f2) or ((f1 == f2) and (self.it > other.it))
# end record

    def __str__(self) -> str:
        """
        Return the string representation of this object.

        :returns: the string representation of this object.
        """
        return f"({self.f}, {self.it})"


def _int_0(_: int) -> int:
    """
    Do nothing.

    :retval 0: always
    """
    return 0


def _float_0() -> float:
    """
    Do nothing.

    :retval 0.0: always
    """
    return 0.0
