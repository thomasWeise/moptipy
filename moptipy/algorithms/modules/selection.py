"""
Selection algorithms are common modules that choose `n` out of `N` objects.

Selection algorithms are modules of the fully-configurable Evolutionary
Algorithm :class:`~moptipy.algorithms.so.general_ea.GeneralEA`. They can
utilize fitness values computed by the fitness assignment processes
(:class:`~moptipy.algorithms.so.fitness.Fitness`). Of course, they can
also be applied in different contexts and are not bound to single-objective
optimization.

:class:`~moptipy.algorithms.modules.selection.Selection` is especially
important in Evolutionary Algorithms
(`~moptipy.algorithms.so.general_ea.GeneralEA`), where it is used in two
places: As *survival selection*, it chooses which points will be allowed to
remain in the population and, hence, survive into the mating pool for the next
generation. As *mating selection* methods, they choose the inputs of the
search operations from the mating pool.

:class:`~moptipy.algorithms.modules.selection.Selection` algorithms must
*only* use the
:attr:`~moptipy.algorithms.modules.selection.FitnessRecord.fitness` of a
solution record (and random numbers) to make their decisions. These fitness
values are subject to minimization. They can equal to the objective values in
optimization or stem from a :class:`~moptipy.algorithms.so.fitness.Fitness`
Assignment Process.

The following selection algorithms have currently been implemented:

- :class:`~moptipy.algorithms.modules.selections.best.Best` selection
  selects the best `n` solutions without replacement. This is a common
  strategy for survival selection, especially in (mu+lambda) EAs
  (compatible to :class:`~moptipy.algorithms.so.ea.EA`).
- :class:`~moptipy.algorithms.modules.selections.random_without_repl\
.RandomWithoutReplacement` selects random solutions without replacement. It is
  a common strategy for mating selection.
- :class:`~moptipy.algorithms.modules.selections.fitness_proportionate_sus\
.FitnessProportionateSUS` performs fitness proportionate selection for
  minimization using stochastic uniform sampling and, optionally, a minimum
  selection probability threshold. It is the classic survival selection
  algorithm in Genetic Algorithm.
- :class:`~moptipy.algorithms.modules.selections.tournament_with_repl.\
TournamentWithReplacement`
  performs tournament selection with a specified tournament size with
  replacement.
- :class:`~moptipy.algorithms.modules.selections.tournament_without_repl.\
TournamentWithoutReplacement`
  performs tournament selection with a specified tournament size without
  replacement.
"""
from typing import Any, Callable, Protocol

from numpy.random import Generator

from moptipy.api.component import Component
from moptipy.utils.types import type_error


# start book
class FitnessRecord(Protocol):
    """A fitness record stores data together with a fitness."""

# end book
    #: the fitness value, the only criterion to be used by a selection
    #: algorithm (besides random numbers)
    fitness: int | float

    def __lt__(self, other) -> bool:
        """
        Compare the fitness of this record with the fitness of another one.

        :param other: the other fitness record
        """
        return self.fitness < other.fitness
# start book


class Selection(Component):
    """The base class for selections algorithms."""

    def select(self, source: list[FitnessRecord],
               dest: Callable[[FitnessRecord], Any],
               n: int, random: Generator) -> None:  # pylint: disable=W0613
        """
        Select `n` records from `source` and pass them to `dest`.

        When choosing the `n` records from `source` to be passed to
        `dest`, only the :attr:`~FitnessRecord.fitness` attribute of
        the records and the random numbers from `random` must be used
        as decision criteria.

        :param source: the list with the records to select from
        :param dest: the destination collector Callable to invoke for
            each selected record, can be :class:`list`.`append`.
        :param n: the number of records to select
        :param random: the random number generator
        """
# end book


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
