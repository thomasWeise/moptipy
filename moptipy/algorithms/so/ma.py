"""
A simple implementation of a (mu+lambda) Memetic Algorithm.

A memetic algorithm (:class:`~moptipy.algorithms.so.ma.MA`) works similar to a
(mu+lambda) :class:`~moptipy.algorithms.so.ea.EA`, but it refines all results
of the search operators with a local search
:attr:`~moptipy.algorithms.so.ma.MA.ls` executed for
:attr:`~moptipy.algorithms.so.ma.MA.ls_fes` objective function evaluations.
It also only employs the nullary search operator (to create the initial random
solutions) and the binary search operator (to combine two selected parental
solutions).

This is the general form of the Memetic Algorithm of the type
"EA+something else." A specialized version combining an
:mod:`~moptipy.algorithms.so.ea` with
:mod:`~moptipy.algorithms.so.rls` can be found in
:mod:`~moptipy.algorithms.so.marls`.

This Memetic Algorithm implementation begins by sampling
:attr:`~moptipy.algorithms.so.ma.MA.mu`
solutions using the nullary search operation
:attr:`~moptipy.api.algorithm.Algorithm0.op0`. Each of these initial solutions
is used as a starting point of a local search
:attr:`~moptipy.algorithms.so.ma.MA.ls`, which is executed for
:attr:`~moptipy.algorithms.so.ma.MA.ls_fes` objective function evaluations.
In each iteration, it then uses the
:attr:`~moptipy.algorithms.so.ma.MA.mu` existing solutions as input for
the binary search operator :attr:`~moptipy.algorithms.so.ma.MA.op2` to create
:attr:`~moptipy.algorithms.so.ma.MA.lambda_` new solutions, each of which is
again used as a starting point of a local search
:attr:`~moptipy.algorithms.so.ma.MA.ls` executed for
:attr:`~moptipy.algorithms.so.ma.MA.ls_fes` objective function evaluations.
The results of the local searches enter the population and in the next
iteration, the :attr:`~moptipy.algorithms.so.ma.MA.mu` best solutions of the
:attr:`~moptipy.algorithms.so.ma.MA.mu` +
:attr:`~moptipy.algorithms.so.ma.MA.lambda_` ones in the population are
retained.

Due to the :class:`~moptipy.api.process.Process` and
:mod:`~moptipy.api.subprocesses` API of `moptipy`, you can use almost arbitrary
algorithms as local search :attr:`~moptipy.algorithms.so.ma.MA.ls`. The only
requirement is that is a subclass of
:class:`~moptipy.api.algorithm.Algorithm0` and uses
it uses an instance of
:class:`moptipy.operators.op0_forward.Op0Forward` as nullary search operator
(:attr:`~moptipy.api.algorithm.Algorithm0.op0`).
This allows the MA to set a solution as starting point for the inner algorithm
:attr:`~moptipy.algorithms.so.ma.MA.ls`.

It should be noted that it is by no means required that the inner algorithm
:attr:`~moptipy.algorithms.so.ma.MA.ls` needs to be a local search. Any
algorithm that fulfills the above requirements could be used. For example, if
we were conducting numerical optimization, it would be totally OK to plug an
instance of the Sequential Least Squares Programming
(:class:`~moptipy.algorithms.so.vector.scipy.SLSQP`) algorithm into the
memetic algorithm&hellip;

Further reading on how to realize using one algorithm as a sub-algorithm
of another one can be found in the documentation of
:func:`~moptipy.api.subprocesses.from_starting_point`,
:func:`~moptipy.api.subprocesses.for_fes`, and
:class:`moptipy.operators.op0_forward.Op0Forward`.

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
"""
from typing import Callable, Final, cast

from numpy.random import Generator

from moptipy.algorithms.so.record import Record
from moptipy.api.algorithm import Algorithm0
from moptipy.api.logging import SCOPE_OP2
from moptipy.api.operators import Op0, Op2, check_op2
from moptipy.api.process import Process
from moptipy.api.subprocesses import for_fes, from_starting_point
from moptipy.operators.op0_forward import Op0Forward
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.strings import PART_SEPARATOR
from moptipy.utils.types import check_int_range, type_error


# start book
class MA(Algorithm0):
    """An MA is a population-based algorithm using binary operators."""

    def solve(self, process: Process) -> None:
        """
        Apply the MA to an optimization problem.

        :param process: the black-box process object
        """
        # initialization of some variables omitted in book for brevity
        # end book
        mu: Final[int] = self.mu  # mu: number of best solutions kept
        mu_plus_lambda: Final[int] = mu + self.lambda_  # size
        random: Final[Generator] = process.get_random()  # random gen
        create: Final[Callable] = process.create  # create x container
        evaluate: Final[Callable] = process.evaluate  # the objective
        op0: Final[Callable] = self.op0.op0  # the nullary operator
        op2: Final[Callable] = self.op2.op2  # the binary operator
        ls_fes: Final[int] = self.ls_fes  # the number of FEs per ls run
        ls_solve: Final[Callable[[Process], None]] = self.ls.solve  # +book
        forward_ls_op0_to: Final[Callable] = cast(  # forward starting
            Op0Forward, self.ls.op0).forward_to  # point of ls to...
        should_terminate: Final[Callable] = process.should_terminate
        r0i: Final[Callable[[int], int]] = cast(  # random integers
            Callable[[int], int], random.integers)
        # start book
        # create list of mu random+ls records and lambda empty records
        lst: Final[list] = [None] * mu_plus_lambda  # pre-allocate list
        f: int | float = 0  # variable to hold objective values
        for i in range(mu_plus_lambda):  # fill list of size mu+lambda
            x = create()  # by creating point in search space
            if i < mu:  # only the first mu records are initialized by
                op0(random, x)  # applying nullary operator = randomize
                if should_terminate():  # should we stop now?
                    cast(Op0Forward, self.ls.op0).stop_forwarding()  # -book
                    return   # computational budget exhausted -> quit
                with for_fes(process, ls_fes) as s1, \
                        from_starting_point(s1, x, evaluate(x)) as s2:
                    forward_ls_op0_to(s2.get_copy_of_best_x)
                    ls_solve(s2)  # apply local search modifying x
                    f = s2.get_best_f()  # get quality of x
            lst[i] = Record(x, f)  # create and store record

        it: int = 0  # set iteration counter=0 (but immediately increment)
        while True:  # lst: keep 0..mu-1, overwrite mu..mu+lambda-1
            it += 1  # step iteration counter
            for oi in range(mu, mu_plus_lambda):  # for all offspring
                if should_terminate():  # should we stop now?
                    cast(Op0Forward, self.ls.op0).stop_forwarding()  # -book
                    return   # computational budget exhausted -> quit
                dest: Record = lst[oi]  # pick destination record
                x = dest.x  # the destination "x" value
                dest.it = it  # remember iteration of solution creation

                sx = lst[r0i(mu)].x  # pick random first source "x"
                sx2 = sx    # second source "x" initially=first sx
                while sx2 is sx:     # until different from sx...
                    sx2 = lst[r0i(mu)].x  # ..get random second "x"
                op2(random, x, sx, sx2)  # apply binary operator
                with for_fes(process, ls_fes) as s1, \
                        from_starting_point(s1, x, evaluate(x)) as s2:
                    forward_ls_op0_to(s2.get_copy_of_best_x)
                    ls_solve(s2)  # apply local search modifying x
                    dest.f = s2.get_best_f()  # get quality of x
            lst.sort()  # best records come first, ties broken by age
            # end book

    def __init__(self, op0: Op0,
                 op2: Op2, ls: Algorithm0,
                 mu: int = 2, lambda_: int = 1,
                 ls_fes: int = 1000,
                 name: str = "ma") -> None:
        """
        Create the Memetic Algorithm (MA).

        :param op0: the nullary search operator
        :param op2: the binary search operator
        :param ls: the local search to apply to each new solution
        :param mu: the number of best solutions to survive in each generation
        :param lambda_: the number of offspring in each generation
        :param ls_fes: the number of FEs (steps) per local search run
        :param name: the base name of the algorithm
        """
        super().__init__(f"{name}{PART_SEPARATOR}{mu}{PART_SEPARATOR}"
                         f"{lambda_}{PART_SEPARATOR}{ls_fes}{PART_SEPARATOR}"
                         f"{op2}{PART_SEPARATOR}{ls}", op0)
        if not isinstance(ls, Algorithm0):
            raise type_error(ls, "ls", Algorithm0)
        if not isinstance(ls.op0, Op0Forward):
            raise type_error(ls.op0, "ls.op0", Op0Forward)
        #: the number of records to survive in each generation
        self.mu: Final[int] = check_int_range(mu, "mu", 1, 1_000_000)
        #: the number of offsprings per generation
        self.lambda_: Final[int] = check_int_range(
            lambda_, "lambda", 1, 1_000_000)
        #: the number of FEs per local search run
        self.ls_fes: Final[int] = check_int_range(
            ls_fes, "ls_fes", 1, 100_000_000)
        #: The binary search operator.
        self.op2: Final[Op2] = check_op2(op2)
        #: the local search algorithm
        self.ls: Final[Algorithm0] = ls

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of the algorithm to a logger.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        logger.key_value("mu", self.mu)
        logger.key_value("lambda", self.lambda_)
        logger.key_value("lsFEs", self.ls_fes)
        with logger.scope(SCOPE_OP2) as o:
            self.op2.log_parameters_to(o)
        with logger.scope("ls") as ls:
            self.ls.log_parameters_to(ls)

    def initialize(self) -> None:
        """Initialize the memetic algorithm."""
        super().initialize()
        self.ls.initialize()
        self.op2.initialize()
