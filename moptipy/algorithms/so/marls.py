"""
A (mu+lambda) Memetic Algorithm using Randomized Local Search (RLS).

A memetic algorithm (:class:`~moptipy.algorithms.so.ma.MA`) works similar to a
(mu+lambda) :class:`~moptipy.algorithms.so.ea.EA`, but it refines all results
of the search operators with a local search, in this case,
:class:`~moptipy.algorithms.so.rls.RLS`, which is executed for
:attr:`~moptipy.algorithms.so.ma.MA.ls_fes` objective function evaluations.
It also only employs the nullary search operator (to create the initial random
solutions) and the binary search operator (to combine two selected parental
solutions), leaving the application of the unary search operator to the RLS.

A general implementation of a Memetic Algorithm into which arbitrary
algorithms can be plugged is given in :mod:`~moptipy.algorithms.so.ma`.
Here, the RLS part and the EA part of the MA are directly merged in a
hard-coded fashion. If the general :class:`~moptipy.algorithms.so.ma.MA` is
configured equivalently to the :class:`~moptipy.algorithms.so.marls.MARLS`
here, i.e., uses the same search operators, same `mu`, `lambda_`, and
`ls_fes`, then both algorithms will do exactly the same search steps.

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
from typing import Any, Callable, Final, cast

from numpy.random import Generator

from moptipy.algorithms.so.record import Record
from moptipy.api.algorithm import Algorithm2
from moptipy.api.operators import Op0, Op1, Op2
from moptipy.api.process import Process
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.strings import PART_SEPARATOR
from moptipy.utils.types import check_int_range, type_error


# start book
class MARLS(Algorithm2):
    """
    A Memetic Algorithm as Evolutionary Algorithm + Randomized Local Search.

    This class implements a Memetic Algorithm (MA) basically as an
    Evolutionary Algorithm (EA), which only uses the nullary and binary search
    operators, and that refines each newly generated solution by applying a
    Randomized Local Search (RLS), which employs the unary search operator.
    """

    def solve(self, process: Process) -> None:
        """
        Apply the MA=EA+RLS to an optimization problem.

        :param process: the black-box process object
        """
        mu: Final[int] = self.mu  # mu: number of best solutions kept
        mu_plus_lambda: Final[int] = mu + self.lambda_  # size
        # initialization of some variables omitted in book for brevity
        # end book
        random: Final[Generator] = process.get_random()  # random gen
        create: Final[Callable] = process.create  # create x container
        evaluate: Final[Callable] = process.evaluate  # the objective
        op0: Final[Callable] = self.op0.op0  # the nullary operator
        op1: Final[Callable] = self.op1.op1  # the unary operator
        op2: Final[Callable] = self.op2.op2  # the binary operator
        ls_fes: Final[int] = self.ls_fes  # the rate at which to use op2
        should_terminate: Final[Callable] = process.should_terminate
        r0i: Final[Callable[[int], int]] = cast(  # random integers
            Callable[[int], int], random.integers)
        copy: Final[Callable[[Any, Any], None]] = process.copy

        # create list of mu random+ls records and lambda empty records
        # start book
        lst: Final[list] = [None] * mu_plus_lambda  # pre-allocate list
        f: int | float = 0  # variable to hold objective values  # -book
        tmp = create()  # the temporary record
        for i in range(mu_plus_lambda):  # fill list of size mu+lambda
            x = create()  # by creating point in search space
            if i < mu:  # only the first mu records are initialized by
                op0(random, x)  # applying nullary operator = randomize
                if should_terminate():  # should we quit?
                    return  # computational budget exhausted -> quit
                f = evaluate(x)  # continue? ok, evaluate new solution
                for _ in range(ls_fes):  # perform ls_fes of RLS
                    op1(random, tmp, x)  # unary operator
                    if should_terminate():  # should we quit?
                        return  # computational budget exhausted
                    ftmp: int | float = evaluate(tmp)  # evaluate new solution
                    if ftmp <= f:  # if it is better or equally good...
                        x, tmp = tmp, x  # accept it via swapping
                        f = ftmp  # and remember quality
            lst[i] = Record(x, f)  # create and store record

        it: int = 0  # set iteration counter=0 (but immediately increment)
        while True:  # lst: keep 0..mu-1, overwrite mu..mu+lambda-1
            it += 1  # step iteration counter
            for oi in range(mu, mu_plus_lambda):  # for all offspring
                if should_terminate():  # only continue if we still...
                    return  # have sufficient budget ... otherwise quit
                dest: Record = lst[oi]  # pick destination record
                x = dest.x  # the destination "x" value
                dest.it = it  # remember iteration of solution creation
                sx2 = sx = lst[r0i(mu)].x  # pick a random source "x"
                while sx2 is sx:  # until different from sx...
                    sx2 = lst[r0i(mu)].x  # ..get random second "x"
                op2(random, x, sx, sx2)  # apply binary operator
                f = evaluate(x)  # evaluate new point
                for _ in range(ls_fes):  # perform ls_fes of RLS
                    op1(random, tmp, x)  # unary operator
                    if should_terminate():  # should we quit?
                        return  # computational budget exhausted
                    ftmp = evaluate(tmp)  # evaluate new solution
                    if ftmp <= f:  # if it is better or equally good...
                        x, tmp = tmp, x  # accept it via swapping {HHH}
                        f = ftmp  # and remember quality
                if x is not dest.x:  # if we had swapped x away at {HHH}
                    copy(dest.x, x)  # store back solution
                    tmp = x  # and put x back into tmp variable
                dest.f = f  # store objective value of refined solution

            lst.sort()  # best records come first, ties broken by age
    # end book

    def __init__(self, op0: Op0,
                 op1: Op1,
                 op2: Op2,
                 mu: int = 1, lambda_: int = 1,
                 ls_fes: int = 1000,
                 name: str = "marls") -> None:
        """
        Create the Memetic Algorithm using hard-coded RLS as local search.

        :param op0: the nullary search operator
        :param op1: the unary search operator
        :param op2: the binary search operator
        :param mu: the number of best solutions to survive in each generation
        :param lambda_: the number of offspring in each generation
        :param ls_fes: the number of FEs (steps) per local search run
        :param name: the base name of the algorithm
        """
        if not isinstance(op0, Op0):
            raise type_error(op0, "op0", Op0)
        if not isinstance(op1, Op1):
            raise type_error(op1, "op1", Op1)
        if not isinstance(op2, Op2):
            raise type_error(op2, "op2", Op2)

        super().__init__(f"{name}{PART_SEPARATOR}{mu}{PART_SEPARATOR}"
                         f"{lambda_}{PART_SEPARATOR}{ls_fes}", op0, op1, op2)
        #: the number of records to survive in each generation
        self.mu: Final[int] = check_int_range(mu, "mu", 1, 1_000_000)
        #: the number of offsprings per generation
        self.lambda_: Final[int] = check_int_range(
            lambda_, "lambda", 1, 1_000_000)
        #: the number of FEs per local search run
        self.ls_fes: Final[int] = check_int_range(
            ls_fes, "ls_fes", 1, 100_000_000)

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of the algorithm to a logger.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        logger.key_value("mu", self.mu)
        logger.key_value("lambda", self.lambda_)
        logger.key_value("lsFEs", self.ls_fes)
