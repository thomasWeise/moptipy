"""
Fitness Assignment Processes assign scalar fitnesses to solutions.

A :class:`~moptipy.algorithms.so.fitness.Fitness` Assignment Process uses
the information of a set of instances of
:class:`~moptipy.algorithms.so.fitness.FRecord` to compute their scalar
:attr:`~moptipy.algorithms.modules.selection.FitnessRecord.fitness`.
This fitness is then used by
:class:`~moptipy.algorithms.modules.selection.Selection` algorithms.
:class:`~moptipy.algorithms.modules.selection.Selection` is important in,
e.g., Evolutionary Algorithms (`~moptipy.algorithms.so.general_ea.GeneralEA`),
where it is used in two places: As *survival selection*, it chooses which
points will be allowed to remain in the population and, hence, survive into
the mating pool for the next generation. As *mating selection* methods, they
choose the inputs of the search operations from the mating pool.

The following :class:`~moptipy.algorithms.so.fitness.Fitness` Assignment
Processes have been implemented so far:

- :class:`~moptipy.algorithms.so.fitnesses.direct.Direct` directly copies the
  objective values (:attr:`~moptipy.algorithms.so.record.Record.f`) of the
  solution records directly over to the fitness.
- :class:`~moptipy.algorithms.so.fitnesses.rank.Rank` ranks the solutions by
  their objective values and uses the ranks as fitness.
- :class:`~moptipy.algorithms.so.fitnesses.rank_and_iteration\
.RankAndIteration` also uses the rank of the objective values in the fitness.
  Additionally, if two solutions have the same objective value but one of them
  is newer, then the newer one will receive the better fitness. This is done
  by accessing the iteration counter
  (:attr:`~moptipy.algorithms.so.record.Record.it`) of the solution records.
- :class:`~moptipy.algorithms.so.fitnesses.ffa.FFA` performs the Frequency
  Fitness Assignment which is suitable for problems with few different
  objective values and large computational budgets.

"""

from math import inf

from numpy.random import Generator

from moptipy.algorithms.modules.selection import FitnessRecord
from moptipy.algorithms.so.record import Record
from moptipy.api.component import Component
from moptipy.api.process import Process
from moptipy.utils.types import type_error


# start book
class FRecord(Record, FitnessRecord):
    """A point `x` in the search space with its quality and fitness."""

# end book

    def __init__(self, x, f: int | float):
        """
        Create the record.

        :param x: the data structure for a point in the search space
        :param f: the corresponding objective value
        """
        super().__init__(x, f)
        #: the fitness assigned to the solution `x`
        self.fitness: int | float = inf

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


# start book
class Fitness(Component):
    """The base class for fitness assignment processes."""

    def assign_fitness(self, p: list[FRecord],
                       random: Generator) -> None:  # pylint: disable=W0613
        """
        Assign a fitness to each element in the list `p`.

        :param p: the list of :class:`FRecord` instances
        :param random: the random number generator
        """
# end book

    def log_information_after_run(self, process: Process) -> None:
        """
        Log the information of this fitness assignment process to the process.

        An instance of :class:`~moptipy.api.process.Process` is given to this
        method after the algorithm has completed its work. The fitness
        assignment process then may store some data as a separate log section
        via :meth:`~moptipy.api.process.Process.add_log_section` if it wants
        to. Implementing this method is optional. This method is only invoked
        if :meth:`~moptipy.api.process.Process.has_log` returns `True`.
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
