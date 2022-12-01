"""
Direct fitness assignment uses the objective values directly as fitness.

Together with fitness proportionate selection (:class:`~moptipy.algorithms.\
modules.selections.fitness_proportionate_sus.FitnessProportionateSUS`), it
turns an EA (:class:`~moptipy.algorithms.so.general_ea.GeneralEA`) to a
minimization variant of the traditional Genetic Algorithms.

1. John Henry Holland. *Adaptation in Natural and Artificial Systems: An
   Introductory Analysis with Applications to Biology, Control, and Artificial
   Intelligence.* Ann Arbor, MI, USA: University of Michigan Press. 1975.
   ISBN: 0-472-08460-7
2. David Edward Goldberg. *Genetic Algorithms in Search, Optimization, and
   Machine Learning.* Boston, MA, USA: Addison-Wesley Longman Publishing Co.,
   Inc. 1989. ISBN: 0-201-15767-5
"""


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
        for rec in p:  # iterate over all records in the population
            rec.fitness = rec.f  # store objective value f as fitness
# end book

    def __str__(self):
        """
        Get the name of this direct fitness assignment.

        :return: the name of this fitness assignment strategy
        :retval "direct": always
        """
        return "direct"
