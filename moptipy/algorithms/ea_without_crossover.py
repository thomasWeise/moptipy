"""
The implementation of an Evolutionary Algorithm without crossover.

This is the basic `mu+lambda`-EA, but without using the binary crossover
operator. It works as follows:

1. Start with a population of `mu` random and `lambda` blank individuals.
2. In each iteration:
    2.1. Retain the best `mu` individuals (which will be the `mu` first
         individuals in the sorted population) and overwrite the `lambda`
         worse ones (which will be at indices `mu...mu+lambda-1`). Each of
         these individuals is overwritten with the results of the unary
         operator applied to one of the `mu` "parents". Each of the `mu`
         parents has the same chance to produce such a new "offspring", but no
         individual can be used as a parent again until all other `mu-1`
         selected individuals have produced at least one offspring. In other
         words, if `lambda > mu`, then each of the `mu` selected individuals
         will produce at least `lambda // mu` offspring and at most
         `1 + lambda // mu`.
    2.2. Shuffle the population to introduce randomness and fairness in the
         case that sorting is stable.
    2.3. Sort the population according to the objective value of the
         individuals. Ties are broken such that younger individuals are
         preferred over older ones (if they have the same objective value).

This EA only applies the unary search operator to sample new points in the
search space. Therefore, its population mainly guards against premature
convergence. If `mu=1` and `lambda=1`, then this algorithm is exactly
equivalent to the :class:`~moptipy.algorithms.rls.RLS` if the same unary and
nullary operator are used. It is only a bit slower due to the additional
overhead of representing a population as list of individual records.
"""
from typing import Final, Union, Callable, List, cast

from numpy.random import Generator

from moptipy.api.algorithm import Algorithm1
from moptipy.api.operators import Op0, Op1
from moptipy.api.process import Process
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.types import type_error
from moptipy.algorithms.utils import Individual, _no_shuffle, _no_random_int


# start book
class EAnoCR(Algorithm1):
    """
    The EA without crossover is a population-based metaheuristic.

    It starts with a population of `mu` random individuals. In each
    step, it retains the `mu` best solutions and generates `lambda`
    new offspring solutions from them using the unary search operator.
    From the joint set of `mu+lambda` solutions, it again selects the
    best `mu` ones for the next iteration. And so on.
    """

    def solve(self, process: Process) -> None:
        """
        Apply the EA to an optimization problem.

        :param process: the black-box process object
        """
        mu: Final[int] = self.__mu  # mu: number of best solutions kept
        pop_size: Final[int] = mu + self.__lambda  # size = mu + lambda
        random: Final[Generator] = process.get_random()  # random gen
        # Omitted for brevity: store function references in variables
        # end book
        create: Final[Callable] = process.create  # create x container
        evaluate: Final[Callable] = process.evaluate  # the objective
        op0: Final[Callable] = self.op0.op0  # the nullary operator
        op1: Final[Callable] = self.op1.op1  # the unary operator
        should_terminate: Final[Callable] = process.should_terminate
        ri: Final[Callable[[int], int]] = cast(    # only if m > 1, we
            Callable[[int], int], random.integers  # need random
            if mu > 1 else _no_random_int)         # indices
        shuffle: Final[Callable[[List], None]] = cast(
            Callable[[List], None], random.shuffle if mu > 1
            else _no_shuffle)  # shuffling only needed if mu > 1
        # start book
        # create population of mu random and lambda empty individuals
        pop: Final[List] = [None] * pop_size  # pre-allocate list
        f: Union[int, float] = 0  # variable to hold objective values
        for i in range(pop_size):  # fill population of size mu+lambda
            x = create()  # by creating point in search space
            if i < mu:  # only the first mu parents are initialized by
                op0(random, x)  # applying nullary operator = randomize
                if should_terminate():  # should we quit?
                    return   # computational budget exhausted -> quit
                f = evaluate(x)  # continue? ok, evaluate new solution
            pop[i] = Individual(x, f)  # create and store record

        gen: int = 1  # The first "real" generation has index 1
        while True:  # pop: keep 0..mu-1, overwrite mu..mu+lambda-1
            end: int = mu  # the end index for parents
            for oi in range(mu, pop_size):  # for all lambda offspring
                if should_terminate():  # only continue if we still...
                    return  # have sufficient budget ... otherwise quit
                offspring: Individual = pop[oi]  # pick offspring
                pi: int = ri(end)  # random parent in 0..end-1 = unused
                parent: Individual = pop[pi]  # pick parent record
                x = offspring.x  # the point in search space we work on
                op1(random, x, parent.x)  # apply unary operator to x
                offspring.f = evaluate(x)  # evaluate new point
                offspring.gen = gen  # mark as member of new generation

                end = end - 1  # decrease number of available parents
                if end != 0:  # swap parent to the end to not use again
                    pop[end], pop[pi] = parent, pop[end]  # parent->old
                else:  # oh: we have used all parents
                    end = mu  # then lambda >= mu and we re-use parents

            gen = gen + 1  # step generation counter

            shuffle(pop)  # ensure total fairness under stable sorting
            pop.sort()  # sort population: best individuals come first
# end book

    def __init__(self, op0: Op0, op1: Op1,
                 mu: int = 1, lambda_: int = 1) -> None:
        """
        Create the Evolutionary Algorithm (EA) without binary crossover.

        :param op0: the nullary search operator
        :param op1: the unary search operator
        :param mu: the number of best solutions to survive in each generation
        :param lambda_: the number of offspring in each generation
        """
        super().__init__(f"eanocr_{mu}_{lambda_}", op0, op1)
        if not isinstance(mu, int):
            raise type_error(mu, "mu", int)
        if not (0 < mu <= 1_000_000):
            raise ValueError(f"invalid mu={mu}, must be in 1..1000000.")
        if not isinstance(lambda_, int):
            raise type_error(lambda_, "lambda", int)
        if not (0 < lambda_ <= 1_000_000):
            raise ValueError(
                f"invalid lambda={lambda_}, must be in 1..1000000.")
        #: the number of individuals to survive in each generation
        self.__mu: Final[int] = mu
        #: the number of offsprings per generation
        self.__lambda: Final[int] = lambda_

    def log_parameters_to(self, logger: KeyValueLogSection):
        """
        Log the parameters of the algorithm to a logger.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        logger.key_value("mu", self.__mu)
        logger.key_value("lambda", self.__lambda)
