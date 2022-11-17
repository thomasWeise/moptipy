"""Use the objective values directly as fitness."""


from numpy.random import Generator

from moptipy.algorithms.so.fitness import Fitness, FRecord


# start book
class Direct(Fitness):
    """
    Objective values are used as fitness directly.

    This is the most primitive and basic fitness assignment strategy.
    It does not give preference to new solutions over old solutions if both
    have the same objective value. In this, it is different from
    :class:`~moptipy.algorithms.so.fitnesses.rank_and_iteration\
.RankAndIteration`, which is the default fitness assignment strategy.
    It is therefore also not compatible to our basic (mu+lambda) Evolutionary
    Algorithm implementation (:class:`~moptipy.algorithms.so.ea.EA`).
    """

    def assign_fitness(self, p: list[FRecord], random: Generator) -> None:
        """
        Assign the objective value as fitness.

        :param p: the list of records
        :param random: ignored
        """
        for rec in p:
            rec.fitness = rec.f
# end book

    def __str__(self):
        """
        Get the name of this direct fitness assignment.

        :return: the name of this fitness assignment strategy
        :retval "direct": always
        """
        return "direct"
