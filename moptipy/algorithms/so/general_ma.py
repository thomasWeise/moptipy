"""
A fully configurable, general (mu+lambda) Memetic Algorithm.

This Memetic Algorithm implementation compares to the one in
:mod:`~moptipy.algorithms.so.ma` like the general Evolutionary Algorithm
from :mod:`~moptipy.algorithms.so.general_ea` compares to the simple one
in :mod:`~moptipy.algorithms.so.ea`: It adds survival and mating selection as
well as a fitness assignment procedure.

It begins by sampling :attr:`~moptipy.algorithms.so.ma.MA.mu`
solutions using the nullary search operation
:attr:`~moptipy.api.algorithm.Algorithm0.op0`. Each of these solutions is
refined for :attr:`~moptipy.algorithms.so.ma.MA.ls_fes` objective function
evaluations using the local search :attr:`~moptipy.algorithms.so.ma.MA.ls`.
In each iteration, this algorithm then uses
:attr:`~moptipy.algorithms.so.ma.MA.mu` existing solutions as input for
the binary search operation :attr:`~moptipy.api.algorithm.Algorithm2.op2`.
The inputs of the operator are chosen from the
:attr:`~moptipy.algorithms.so.ma.MA.mu` solutions using
:attr:`~moptipy.algorithms.so.general_ma.GeneralMA.mating`
selection. Each of the :attr:`~moptipy.algorithms.so.ma.MA.lambda_` new
solutions have been created this way are again refined for
:attr:`~moptipy.algorithms.so.ma.MA.ls_fes` objective function evaluations
using the local search :attr:`~moptipy.algorithms.so.ma.MA.ls`. Then, a
fitness assignment process (:class:`~moptipy.algorithms.so.fitness.Fitness`)
assigns fitness values to them based on their objective values
(:attr:`~moptipy.algorithms.so.record.Record.f`), maybe also using the index
of the iteration (:attr:`~moptipy.algorithms.so.record.Record.it`) in which
they were created. The survival selection
:attr:`~moptipy.algorithms.so.general_ma.GeneralMA.survival` then chooses,
from the joint set of `mu+lambda` solutions, the `mu` solutions for the
next iteration. Both mating and survival selection are instances of class
:class:`~moptipy.algorithms.modules.selection.Selection`.

1. Pablo Moscato. *On Evolution, Search, Optimization, Genetic Algorithms and
   Martial Arts: Towards Memetic Algorithms.* Caltech Concurrent Computation
   Program Report C3P 826. 1989. Pasadena, CA, USA: California Institute of
   Technology (Caltech), Caltech Concurrent Computation Program (C3P).
   https://www.researchgate.net/publication/2354457
2. Carlos Cotta, Luke Mathieson, and Pablo Moscato. Memetic Algorithms. In
   Rafael Martí, Panos M. Pardalos, and Mauricio G. C. Resende, editors,
   *Handbook of Heuristics.* Part~III: Metaheuristics, pages 607-638. 2018.
   Cham, Switzerland: Springer. ISBN: 978-3-319-07123-7.
   doi: https://doi.org/10.1007/978-3-319-07153-4_29-1
   https://www.researchgate.net/publication/315660932
3. William Eugene Hart, James E. Smith, and Natalio Krasnogor, editors.
   *Recent Advances in Memetic Algorithms.* Studies in Fuzziness and Soft
   Computing (STUDFUZZ), volume 166. 2005. Berlin/Heidelberg, Germany:
   Springer. ISBN: 978-3-540-22904-9.
   doi: https://doi.org/10.1007/3-540-32363-5
4. Ferrante Neri, Carlos Cotta, and Pablo Moscato. *Handbook of Memetic
   Algorithms.* Volume 379 of Studies in Computational Intelligence (SCI).
   2012. Berlin/Heidelberg, Germany: Springer. ISBN: 978-3-642-23246-6
   doi https://doi.org/10.1007/978-3-642-23247-3.
5. L. Darrell Whitley, V. Scott Gordon, and Keith E. Mathias. Lamarckian
   Evolution, The Baldwin Effect and Function Optimization. In Yuval Davidor,
   Hans-Paul Schwefel, and Reinhard Männer, editors, *Proceedings of the Third
   Conference on Parallel Problem Solving from Nature; International
   Conference on Evolutionary Computation (PPSN III),* October 9-14, 1994,
   Jerusalem, Israel, pages 5-15. Volume 866/1994 of Lecture Notes in Computer
   Science, Berlin, Germany: Springer-Verlag GmbH. ISBN 0387584846.
   doi: https://doi.org/10.1007/3-540-58484-6_245.
   https://www.researchgate.net/publication/2521727
6. Thomas Bäck, David B. Fogel, and Zbigniew Michalewicz, eds., *Handbook of
   Evolutionary Computation.* 1997. Computational Intelligence Library.
   New York, NY, USA: Oxford University Press, Inc. ISBN: 0-7503-0392-1
7. James C. Spall. *Introduction to Stochastic Search and Optimization.*
   Estimation, Simulation, and Control - Wiley-Interscience Series in Discrete
   Mathematics and Optimization, volume 6. 2003. Chichester, West Sussex, UK:
   Wiley Interscience. ISBN: 0-471-33052-3. http://www.jhuapl.edu/ISSO/.
8. Frank Hoffmeister and Thomas Bäck. Genetic Algorithms and Evolution
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
from moptipy.algorithms.so.fitness import Fitness, FRecord, check_fitness
from moptipy.algorithms.so.fitnesses.rank_and_iteration import RankAndIteration
from moptipy.algorithms.so.general_ea import _Record
from moptipy.algorithms.so.ma import MA
from moptipy.api.algorithm import Algorithm0
from moptipy.api.operators import Op0, Op2
from moptipy.api.process import Process
from moptipy.api.subprocesses import for_fes, from_starting_point
from moptipy.operators.op0_forward import Op0Forward
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.strings import PART_SEPARATOR


class GeneralMA(MA):
    """The fully customizable (mu+lambda) MA."""

    def solve(self, process: Process) -> None:
        """
        Apply the (mu+lambda) MA to an optimization problem.

        :param process: the black-box process object
        """
        # initialization of some variables omitted in book for brevity
        mu: Final[int] = self.mu  # mu: number of best solutions kept
        lambda_: Final[int] = self.lambda_  # number of new solutions/gen
        mu_plus_lambda: Final[int] = mu + lambda_  # size = mu + lambda
        random: Final[Generator] = process.get_random()  # random gen
        create: Final[Callable] = process.create  # create x container
        evaluate: Final[Callable] = process.evaluate  # the objective
        op0: Final[Callable] = self.op0.op0  # the nullary operator
        op2: Final[Callable] = self.op2.op2  # the binary operator
        should_terminate: Final[Callable] = process.should_terminate
        ls_fes: Final[int] = self.ls_fes  # the number of FEs per ls run
        ls_solve: Final[Callable[[Process], None]] = self.ls.solve  # +book
        forward_ls_op0_to: Final[Callable] = cast(  # forward starting
            Op0Forward, self.ls.op0).forward_to  # point of ls to...
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

        # create list of mu random/ls records and lambda empty records
        f: int | float = 0  # variable to hold objective values
        for i in range(mu_plus_lambda):  # fill list of size mu+lambda
            x = create()  # by creating point in search space
            selected: bool = i < mu  # only fully create first mu recs
            if selected:  # only the first mu records are initialized by
                op0(random, x)  # applying nullary operator = randomize
                if should_terminate():  # should we stop now?
                    cast(Op0Forward, self.ls.op0).stop_forwarding()
                    if process.has_log():
                        self.fitness.log_information_after_run(process)
                    return   # computational budget exhausted -> quit
                with for_fes(process, ls_fes) as s1, \
                        from_starting_point(s1, x, evaluate(x)) as s2:
                    forward_ls_op0_to(s2.get_copy_of_best_x)
                    ls_solve(s2)  # apply local search modifying x
                    f = s2.get_best_f()  # get quality of x
            recs[i] = _Record(x, f, selected)  # create and store record

        mating_pool: Final[list] = recs[0:mu]  # the selection survivors
        assign_fitness(mating_pool, random)  # assign fitness first time

        mating_pool_clear: Final[Callable[[], None]] = mating_pool.clear
        mating_pool_append: Final[Callable[[FitnessRecord], None]] = \
            cast(Callable[[FitnessRecord], None], mating_pool.append)

        it: int = 0  # set the iteration counter
        while True:  # lst: keep 0..mu-1, overwrite mu..mu+lambda-1
            it = it + 1  # step the iteration counter
            population_clear()  # clear population

            di = 0  # set index of next potential destination
            for _ in range(lambda_):  # for all lambda offspring
                if should_terminate():  # should we stop now?
                    cast(Op0Forward, self.ls.op0).stop_forwarding()
                    if process.has_log():
                        self.fitness.log_information_after_run(process)
                    return   # computational budget exhausted -> quit
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
                parents_clear()  # clear mating pool to make room for 2
                mating_selection(mating_pool, parents_append, 2, random)

                op2(random, x, parents[0].x, parents[1].x)
                with for_fes(process, ls_fes) as s1, \
                        from_starting_point(s1, x, evaluate(x)) as s2:
                    forward_ls_op0_to(s2.get_copy_of_best_x)
                    ls_solve(s2)  # apply local search modifying x
                    dest.f = s2.get_best_f()  # get quality of x

                population_append(dest)  # store in population

            # add remaining selected solutions from recs to population
            for di2 in range(di, mu_plus_lambda):
                other = recs[di2]
                if other._selected:  # only if solution was selected
                    other._selected = False  # set as unselected
                    population_append(other)  # put into population

            assign_fitness(population, random)  # assign fitness
            mating_pool_clear()  # clear list of survived records
            survival_selection(population, mating_pool_append, mu, random)
            for rec in mating_pool:  # mark all selected solutions as
                rec._selected = True  # selected

    def __init__(self, op0: Op0, op2: Op2,
                 ls: Algorithm0,
                 mu: int = 2, lambda_: int = 1,
                 ls_fes: int = 1000,
                 fitness: Fitness | None = None,
                 survival: Selection | None = None,
                 mating: Selection | None = None,
                 name: str = "generalMa") -> None:
        """
        Create the customizable Memetic Algorithm (MA).

        :param op0: the nullary search operator
        :param op2: the binary search operator
        :param ls: the local search to apply to each new solution
        :param mu: the number of best solutions to survive in each generation
        :param lambda_: the number of offspring in each generation
        :param ls_fes: the number of FEs (steps) per local search run
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

        super().__init__(op0, op2, ls, mu, lambda_, ls_fes, name)
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
