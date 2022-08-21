"""The base class for fitness assignment processes."""

from math import inf
from typing import Union, Final, List

from numpy.random import Generator

from moptipy.api.component import Component
from moptipy.utils.types import type_error
from moptipy.algorithms.so.record import Record


class FRecord(Record):
    """A point `x` in the search space with its quality and fitness."""

    def __init__(self, x, f: Union[int, float]):
        """
        Create the record.

        :param x: the data structure for a point in the search space
        :param f: the corresponding objective value
        """
        super().__init__(x, f)
        #: the fitness assigned to the solution `x`
        self.v: Union[int, float] = inf

    def __lt__(self, other) -> bool:
        """
        Precedence if 1) better ftness or b) equally good but younger.

        :param other: the other record
        :returns: `True` if this record has a better fitness value
            (:attr:`v`) or if it has the same objective value but is newer,
            i.e., has a larger :attr:`~moptipy.algorithms.so.record.Record.it`
            value

        >>> r1 = FRecord(None, 1)
        >>> r2 = FRecord(None, 1)
        >>> r1 < r2
        False
        >>> r2 < r1
        False
        >>> r1.v = 10
        >>> r2.v = 9
        >>> r2 < r1
        True
        >>> r1 < r2
        False
        >>> r1.v = r2.v
        >>> r1.it = 10
        >>> r2.it = 9
        >>> r1 < r2
        True
        >>> r2 < r1
        False
        """
        v1: Final[Union[int, float]] = self.v
        v2: Final[Union[int, float]] = other.v
        return (v1 < v2) or ((v1 == v2) and (self.it > other.it))


class Fitness(Component):
    """The base class for fitness assignment processes."""

    def assign_fitness(self,
                       p: List[FRecord],
                       random: Generator) -> None:  # pylint: disable=W0613
        """
        Assign a fitness to each element in the list `p`.

        :param p: the list of :class:`FRecord` instances
        :param random: the random number generator
        """
        for i in p:
            i.v = i.f

    def __str__(self):
        """
        Get the name of the fitness assignment process.

        :return: the name of the fitness assignment process
        """
        return "f"


def check_fitness(fitness: Fitness) -> Fitness:
    """
    Check whether an object is a valid instance of :class:`Fitness`.

    :param fitness: the Fitness object
    :return: the object
    :raises TypeError: if `fitness` is not an instance of :class:`Fitness`
    """
    if not isinstance(fitness, Fitness):
        raise type_error(fitness, "op0", Fitness)
    return fitness
