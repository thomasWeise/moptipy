"""A simple record for storing a solution with its quality."""

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
    If the population is sorted, then records with better objective
    value will be moved to the beginning of the list. Ties are broken
    such that younger individuals (with higher :attr:`it` value) are
    preferred.
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

    def __lt__(self, other) -> bool:
        """
        Precedence if 1) better or b) equally good but younger.

        :param other: the other record
        :returns: `True` if this record has a better objective value
            (:attr:`f`) or if it has the same objective value but is newer,
            i.e., has a larger :attr:`it` value

        >>> r1 = Record(None, 10)
        >>> r2 = Record(None, 9)
        >>> r1 < r2
        False
        >>> r2 < r1
        True
        >>> r1.it = 22
        >>> r2.f = r1.f
        >>> r2.it = r1.it
        >>> r1 < r2
        False
        >>> r2 < r1
        False
        >>> r2.it = r1.it + 1
        >>> r1 < r2
        False
        >>> r2 < r1
        True
        """
        f1: Final[Union[int, float]] = self.f
        f2: Final[Union[int, float]] = other.f
        return (f1 < f2) or ((f1 == f2) and (self.it > other.it))
# end record
