"""The base class for selection algorithms."""
from typing import List, Protocol, Union

from numpy.random import Generator

from moptipy.api.component import Component
from moptipy.utils.types import type_error


class FitnessRecord(Protocol):
    """
    A fitness record stores data together with a fitness.

    The fitness should then be the only criterion used for selection.
    """

    #: the fitness value
    fitness: Union[int, float]

    def __lt__(self, other) -> bool:
        """
        Compare this fitness record with another fitness record.

        :param other: the other fitness record
        """
        return self.fitness < other.fitness


class Selection(Component):
    """The base class for selections algorithms."""

    def select(self, source: List[FitnessRecord], dest: List[FitnessRecord],
               n: int, random: Generator) -> None:  # pylint: disable=W0613
        """
        Select `n` records from `source` and append them to `dest`.

        When choosing the `n` records from `source` to be appended to `dest`,
        only the :attr:`~FitnessRecord.fitness` attribute of the records and
        the random numbers from `random` should be used as decision criteria.

        Selection algorithms are modules of the fully-configurable
        Evolutionary Algorithm :class:`~moptipy.algorithms.so.full_ea.FullEA`.
        They can utilize fitness values computed by the fitness assignment
        processes (:class:`~moptipy.algorithms.so.fitness.Fitness`). Of
        course, they can also be applied in different contexts and are not
        bound to single-objective optimization.

        :param source: the list with the records to select from
        :param dest: the destination to append the selected records to
        :param n: the number of records to select
        :param random: the random number generator
        """


def check_selection(selection: Selection) -> Selection:
    """
    Check whether an object is a valid instance of :class:`Selection`.

    :param selection: the Selection object
    :return: the object
    :raises TypeError: if `selections` is not an instance of
        :class:`Selection` or if it is an instance of the abstract base
        class
    """
    if not isinstance(selection, Selection):
        raise type_error(selection, "selections", Selection)
    if selection.__class__ is Selection:
        raise TypeError("cannot use abstract class 'Selection' directly")
    return selection
