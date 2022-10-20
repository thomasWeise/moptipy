"""The base class for fitness assignment processes."""

from math import inf
from typing import Union, List

from numpy.random import Generator

from moptipy.algorithms.modules.selection import FitnessRecord
from moptipy.algorithms.so.record import Record
from moptipy.api.component import Component
from moptipy.utils.types import type_error


class FRecord(Record, FitnessRecord):
    """A point `x` in the search space with its quality and fitness."""

    def __init__(self, x, f: Union[int, float]):
        """
        Create the record.

        :param x: the data structure for a point in the search space
        :param f: the corresponding objective value
        """
        super().__init__(x, f)
        #: the fitness assigned to the solution `x`
        self.fitness: Union[int, float] = inf

    def __lt__(self, other) -> bool:
        """
        Precedence on better ftness.

        :param other: the other record
        :returns: `True` if this record has a better fitness value
            (:attr:`fitness`)

        >>> r1 = FRecord(None, 1)
        >>> r2 = FRecord(None, 1)
        >>> r1 < r2
        False
        >>> r2 < r1
        False
        >>> r1.fitness = 10
        >>> r2.fitness = 9
        >>> r2 < r1
        True
        >>> r1 < r2
        False
        >>> r1.fitness = r2.fitness
        >>> r1.it = 10
        >>> r2.it = 9
        >>> r1 < r2
        False
        >>> r2 < r1
        False
        """
        return self.fitness < other.fitness


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


def check_fitness(fitness: Fitness) -> Fitness:
    """
    Check whether an object is a valid instance of :class:`Fitness`.

    :param fitness: the Fitness object
    :return: the object
    :raises TypeError: if `fitness` is not an instance of :class:`Fitness`
    """
    if not isinstance(fitness, Fitness):
        raise type_error(fitness, "op0", Fitness)
    if fitness.__class__ is Fitness:
        raise TypeError("cannot use abstract class 'Fitness' directly")
    return fitness
