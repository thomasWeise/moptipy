"""
A fully configurable, general (mu+lambda) Evolutionary Algorithm.

This evolutionary algorithm begins by sampling
:attr:`~moptipy.algorithms.so.ea.EA.mu`
solutions using the nullary search operation
:attr:`~moptipy.api.algorithm.Algorithm0.op0`. In each iteration, it then uses
:attr:`~moptipy.algorithms.so.ea.EA.mu` existing solutions as input for
the search operations, where, for each solution to be sampled, the binary
operation :attr:`~moptipy.api.algorithm.Algorithm2.op2` is used with
probability :attr:`~moptipy.algorithms.so.ea.EA.br` and (otherwise), the unary
operator :attr:`~moptipy.api.algorithm.Algorithm1` is used. The inputs of both
operators are chosen from the :attr:`~moptipy.algorithms.so.ea.EA.mu`
solutions using :attr:`~moptipy.algorithms.so.general_ea.GeneralEA.mating`
selection. After :attr:`~moptipy.algorithms.so.ea.EA.lambda_` new solutions
have been created this way (and have been evaluated as well), a fitness
assignment process (:class:`~moptipy.algorithms.so.fitness.Fitness`) assigns
fitness values to them based on their objective values
(:attr:`~moptipy.algorithms.so.record.Record.f`), maybe also using the index
of the iteration (:attr:`~moptipy.algorithms.so.record.Record.it`) in which
they were created. The survival selection
:attr:`~moptipy.algorithms.so.general_ea.GeneralEA.survival` then chooses,
from the joint set of `mu+lambda` solutions, the `mu` solutions for the
next iteration. Both mating and survival selection are instances of class
:class:`~moptipy.algorithms.modules.selection.Selection`.

This algorithm is equivalent to :class:`~moptipy.algorithms.so.ea.EA`, but
allows for using a customized fitness assignment step
(:class:`~moptipy.algorithms.so.fitness.Fitness`) as well as customizable
survival and :attr:`~moptipy.algorithms.so.general_ea.GeneralEA.mating`
selection (:class:`~moptipy.algorithms.modules.selection.Selection`).

1. Thomas Bäck, David B. Fogel, and Zbigniew Michalewicz, eds., *Handbook of
   Evolutionary Computation.* 1997. Computational Intelligence Library.
   New York, NY, USA: Oxford University Press, Inc. ISBN: 0-7503-0392-1
2. James C. Spall. *Introduction to Stochastic Search and Optimization.*
   Estimation, Simulation, and Control - Wiley-Interscience Series in Discrete
   Mathematics and Optimization, volume 6. 2003. Chichester, West Sussex, UK:
   Wiley Interscience. ISBN: 0-471-33052-3. http://www.jhuapl.edu/ISSO/.
3. Frank Hoffmeister and Thomas Bäck. Genetic Algorithms and Evolution
   Strategies: Similarities and Differences. In Hans-Paul Schwefel and
   Reinhard Männer, *Proceedings of the International Conference on Parallel
   Problem Solving from Nature (PPSN I),* October 1-3, 1990, Dortmund,
   Germany, volume 496 of Lecture Notes in Computer Science, pages 455-469,
   Berlin/Heidelberg, Germany: Springer. ISBN: 978-3-540-54148-6.
   https://doi.org/10.1007/BFb0029787.
"""
from typing import Callable, Final, cast

from numpy.random import Generator

from moptipy.algorithms.modules.selection import (
    FitnessRecord,
    Selection,
    check_selection,
)
from moptipy.algorithms.modules.selections.best import Best
from moptipy.algorithms.modules.selections.random_without_repl import (
    RandomWithoutReplacement,
)
from moptipy.algorithms.so.ea import EA, _float_0
from moptipy.algorithms.so.fitness import Fitness, FRecord, check_fitness
from moptipy.algorithms.so.fitnesses.rank_and_iteration import RankAndIteration
from moptipy.api.operators import Op0, Op1, Op2
from moptipy.api.process import Process
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.strings import PART_SEPARATOR


class _Record(FRecord):
    """Same as :class:`FRecord`, but with a secret selection marker."""

    def __init__(self, x, f: int | float, selected: bool = False):
        """
        Create the record.

        :param x: the data structure for a point in the search space
        :param f: the corresponding objective value
        :param selected: is the record currently in use?
        """
        super().__init__(x, f)
        #: an internal flag - do NOT access!!
        self._selected: bool = selected


# start book
class GeneralEA(EA):
    """The fully customizable (mu+lambda) EA."""

    def solve(self, process: Process) -> None:
        """
        Apply the (mu+lambda) EA to an optimization problem.

        :param process: the black-box process object
        """
        # initialization of some variables omitted in book for brevity
# end book
        mu: Final[int] = self.mu  # mu: number of best solutions kept
        lambda_: Final[int] = self.lambda_  # number of new solutions/gen
        mu_plus_lambda: Final[int] = mu + lambda_  # size = mu + lambda
        random: Final[Generator] = process.get_random()  # random gen
        create: Final[Callable] = process.create  # create x container
        evaluate: Final[Callable] = process.evaluate  # the objective
        op0: Final[Callable] = self.op0.op0  # the nullary operator
        op1: Final[Callable] = self.op1.op1  # the unary operator
        op2: Final[Callable] = self.op2.op2  # the binary operator
        br: Final[float] = self.br  # the rate at which to use op2
        should_terminate: Final[Callable] = process.should_terminate
        r01: Final[Callable[[], float]] = cast(  # only if 0<br<1, we
            Callable[[], float],  # need random floats
            random.random if 0 < br < 1 else _float_0)
        assign_fitness: Final[Callable[[list[FRecord], Generator], None]] = \
            self.fitness.assign_fitness
        survival_selection: Final[Callable[
            [list[FRecord], Callable, int, Generator], None]] = \
            cast(Callable[[list[FRecord], Callable, int, Generator],
                          None], self.survival.select)
        mating_selection: Final[Callable[
            [list[FRecord], Callable, int, Generator], None]] = \
            cast(Callable[[list[FRecord], Callable, int, Generator],
                          None], self.mating.select)
        recs: Final[list] = [None] * mu_plus_lambda  # pre-allocate list
        parents: Final[list] = [None, None]  # mating pool: length 2
        population: Final[list] = [None] * mu_plus_lambda  # whole pop
        parents_clear: Final[Callable[[], None]] = parents.clear
        parents_append: Final[Callable[[FitnessRecord], None]] = \
            cast(Callable[[FitnessRecord], None], parents.append)
        population_clear: Final[Callable[[], None]] = population.clear
        population_append: Final[Callable[[_Record], None]] = \
            cast(Callable[[_Record], None], population.append)
# start book
        # create list of mu random records and lambda empty records
        f: int | float = 0  # variable to hold objective values
        for i in range(mu_plus_lambda):  # fill list of size mu+lambda
            x = create()  # by creating point in search space
            selected: bool = i < mu  # only fully create first mu recs
            if selected:  # only the first mu records are initialized by
                op0(random, x)  # applying nullary operator = randomize
                if should_terminate():  # should we quit?  # -book
                    if process.has_log():  # -book
                        self.fitness.log_information_after_run(  # -book
                            process)  # -book
                    return  # computational budget exhausted -> quit  # -book
                f = evaluate(x)  # continue? ok, evaluate new solution
            recs[i] = _Record(x, f, selected)  # create and store record

        mating_pool: Final[list] = recs[0:mu]  # the selection survivors
        assign_fitness(mating_pool, random)  # assign fitness first time
# end book
        mating_pool_clear: Final[Callable[[], None]] = mating_pool.clear
        mating_pool_append: Final[Callable[[FitnessRecord], None]] = \
            cast(Callable[[FitnessRecord], None], mating_pool.append)
# start book
        it: int = 0  # set the iteration counter
        while True:  # lst: keep 0..mu-1, overwrite mu..mu+lambda-1
            it = it + 1  # step the iteration counter
            population_clear()  # clear population

            di = 0  # set index of next potential destination
            for _ in range(lambda_):  # for all lambda offspring
                if should_terminate():  # only continue if we still... # -book
                    if process.has_log():  # -book
                        self.fitness.log_information_after_run(  # -book
                            process)  # -book
                    return  # ...have sufficient budget # -book
                while True:  # get the next non-selected record
                    dest = recs[di]  # get the record
                    di = di + 1  # step counter
                    if dest._selected:  # if it was selected
                        dest._selected = False  # mark it as unselected
                        population_append(dest)  # store in population
                        continue  # try next record
                    break  # use the (unselected) record as destination

                x = dest.x  # the destination "x" value
                dest.it = it  # remember iteration of solution creation
                do_binary: bool = r01() < br  # will we do binary operation?
                parents_clear()  # clear mating pool: room for 2
                mating_selection(mating_pool, parents_append,
                                 2 if do_binary else 1, random)

                if do_binary:  # binary operation (with p == br)
                    op2(random, x, parents[0].x, parents[1].x)
                else:  # unary operation otherwise
                    op1(random, x, parents[0].x)  # apply unary op
                dest.f = evaluate(x)  # evaluate new point
                population_append(dest)  # store in population

            # add remaining selected solutions from recs to population
            # from index di to mu+lambda ... omitted for brevity in book
            # end book
            for di2 in range(di, mu_plus_lambda):
                other = recs[di2]
                if other._selected:  # only if solution was selected
                    other._selected = False  # set as unselected
                    population_append(other)  # put into population
            # start book
            assign_fitness(population, random)  # assign fitness
            mating_pool_clear()  # clear list of survived records
            survival_selection(population, mating_pool_append, mu, random)
            for rec in mating_pool:  # mark all selected solutions as
                rec._selected = True  # selected
# end book

    def __init__(self, op0: Op0,
                 op1: Op1 | None = None,
                 op2: Op2 | None = None,
                 mu: int = 1, lambda_: int = 1,
                 br: float | None = None,
                 fitness: Fitness | None = None,
                 survival: Selection | None = None,
                 mating: Selection | None = None,
                 name: str = "generalEa") -> None:
        """
        Create the customizable Evolutionary Algorithm (EA).

        :param op0: the nullary search operator
        :param op1: the unary search operator
        :param op2: the binary search operator
        :param mu: the number of best solutions to survive in each generation
        :param lambda_: the number of offspring in each generation
        :param br: the rate at which the binary operator is applied
        :param fitness: the fitness assignment process
        :param survival: the survival selections algorithm
        :param mating: the mating selections algorithm
        :param name: the base name of the algorithm
        """
        if fitness is None:
            fitness = RankAndIteration()
        if fitness.__class__ is not RankAndIteration:
            name = f"{name}{PART_SEPARATOR}{fitness}"
        if survival is None:
            survival = Best()
        if mating is None:
            mating = RandomWithoutReplacement()
        if (survival.__class__ is not Best) \
                or (mating.__class__ is not RandomWithoutReplacement):
            name = f"{name}{PART_SEPARATOR}{survival}{PART_SEPARATOR}{mating}"

        super().__init__(op0, op1, op2, mu, lambda_, br, name)
        #: the fitness assignment process
        self.fitness: Final[Fitness] = check_fitness(fitness)
        #: the survival selection algorithm
        self.survival: Final[Selection] = check_selection(survival)
        #: the mating selection algorithm
        self.mating: Final[Selection] = check_selection(mating)

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of the algorithm to a logger.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        with logger.scope("fitness") as v:
            self.fitness.log_parameters_to(v)
        with logger.scope("survival") as s:
            self.survival.log_parameters_to(s)
        with logger.scope("mating") as m:
            self.mating.log_parameters_to(m)

    def initialize(self) -> None:
        """Initialize the algorithm."""
        super().initialize()
        self.survival.initialize()
        self.mating.initialize()
        self.fitness.initialize()
