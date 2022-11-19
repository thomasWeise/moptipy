"""
A simple implementation of a (mu+lambda) Evolutionary Algorithm.

This is the basic `mu+lambda`-EA, which works as follows:

1. Start with a list `lst` of `mu` random records and `lambda` blank records.
2. In each iteration:

    2.1. Use the `mu` first records as input to the search operators to
         generate `lambda` new points in the search space.
         For each new point to be created, the binary operator is applied
         with probability `0<=br<=1` and the unary operator is used otherwise.

    2.2. Sort the list `lst` according to the objective value of the record.
         Ties are broken by preferring younger solutions over old ones. Soring
         uses the `__lt__` dunder method of class
         :class:`~moptipy.algorithms.so.record.Record`. This moves the best
         solutions to the front of the list. The tie breaking method both
         encourages drift and ensures compatibility with `RLS`.

If `mu=1`, `lambda=1`, and `br=0`, then this algorithm is exactly equivalent
to the :class:`~moptipy.algorithms.so.rls.RLS` if the same unary and nullary
operator are used. It is only a bit slower due to the additional overhead of
maintaining a list of records. This compatibility is achieved by the tie
breaking strategy of `step 2.2` above: RLS will prefer the newer solution over
the current one if the new solution is either better or as same as good. Now
the latter case cannot be achieved by just sorting the list without
considering the iteration at which a solution was created, since sorting in
Python is *stable* (equal elements remain in the order in which they are
encountered in the original list) and because our new solutions would be in
the `lambda` last entries of the list. This can easily be fixed by the tie
breaking, which is implemented in the `__lt__` dunder method of class
:class:`~moptipy.algorithms.so.record.Record`.

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
from math import isfinite
from typing import Callable, Final, cast

from numpy.random import Generator

from moptipy.algorithms.so.record import Record
from moptipy.api.algorithm import Algorithm2
from moptipy.api.operators import Op0, Op1, Op2
from moptipy.api.process import Process
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.strings import PART_SEPARATOR, num_to_str_for_name
from moptipy.utils.types import type_error


def _int_0(_: int) -> int:
    """
    Return an integer with value `0`.

    :retval 0: always
    """
    return 0


def _float_0() -> float:
    """
    Return a float with value `0.0`.

    :retval 0.0: always
    """
    return 0.0


# start nobinary
class EA(Algorithm2):
    """
    The EA is a population-based algorithm using unary and binary operators.

    It starts with a list of :attr:`mu` randomly initialized solutions. In
    each step, it retains the `mu` best solutions and generates
    :attr:`lambda_` new solutions from them using the unary operator
    (:attr:`~moptipy.api.algorithm.Algorithm1.op1`) with probability
    1-:attr:`br` and the binary search operator
    ((:attr:`~moptipy.api.algorithm.Algorithm2.op2`) at rate :attr:`br`. From
    the joint set of `mu+lambda_` solutions, it again selects the best `mu`
    ones for the next iteration. And so on.

    If `mu=1`, `lambda_=1`, and `br=0`, then this algorithm is exactly
    equivalent to the :class:`~moptipy.algorithms.so.rls.RLS` if the same
    unary and nullary operator are used.
    """

    def solve(self, process: Process) -> None:
        """
        Apply the EA to an optimization problem.

        :param process: the black-box process object
        """
        mu: Final[int] = self.mu  # mu: number of best solutions kept
        mu_plus_lambda: Final[int] = mu + self.lambda_  # size
        # Omitted for brevity: store function references in variables
        # end nobinary
        random: Final[Generator] = process.get_random()  # random gen
        create: Final[Callable] = process.create  # create x container
        evaluate: Final[Callable] = process.evaluate  # the objective
        op0: Final[Callable] = self.op0.op0  # the nullary operator
        op1: Final[Callable] = self.op1.op1  # the unary operator
        op2: Final[Callable] = self.op2.op2  # the binary operator
        br: Final[float] = self.br  # the rate at which to use op2
        should_terminate: Final[Callable] = process.should_terminate
        r0i: Final[Callable[[int], int]] = cast(   # only if m > 1, we
            Callable[[int], int], random.integers  # need random
            if mu > 1 else _int_0)                 # indices
        r01: Final[Callable[[], float]] = cast(  # only if 0<br<1, we
            Callable[[], float],                 # need random floats
            random.random if 0 < br < 1 else _float_0)
        # start nobinary
        # create list of mu random records and lambda empty records
        lst: Final[list] = [None] * mu_plus_lambda  # pre-allocate list
        f: int | float = 0  # variable to hold objective values
        for i in range(mu_plus_lambda):  # fill list of size mu+lambda
            x = create()  # by creating point in search space
            if i < mu:  # only the first mu records are initialized by
                op0(random, x)  # applying nullary operator = randomize
                if should_terminate():  # should we quit?
                    return   # computational budget exhausted -> quit
                f = evaluate(x)  # continue? ok, evaluate new solution
            lst[i] = Record(x, f)  # create and store record

        it: int = 0
        while True:  # lst: keep 0..mu-1, overwrite mu..mu+lambda-1
            it += 1  # step iteration counter
            for oi in range(mu, mu_plus_lambda):  # for all offspring
                if should_terminate():  # only continue if we still...
                    return  # have sufficient budget ... otherwise quit
                dest: Record = lst[oi]  # pick destination record
                x = dest.x  # the destination "x" value
                dest.it = it  # remember iteration of solution creation

                sx = lst[r0i(mu)].x  # pick a random source record
                # end nobinary
                # start binary
                if r01() < br:  # apply binary operator at rate br
                    sx2 = sx    # second source "x"
                    while sx2 is sx:     # must be different from sx
                        sx2 = lst[r0i(mu)].x  # get second record
                    op2(random, x, sx, sx2)  # apply binary op
                    dest.f = evaluate(x)  # evaluate new point
                    continue  # below is "else" part with unary operat.
                # end binary
                # start nobinary
                op1(random, x, sx)  # apply unary operator
                dest.f = evaluate(x)  # evaluate new point

            lst.sort()  # best records come first, ties broken by age
# end nobinary

    def __init__(self, op0: Op0,
                 op1: Op1 | None = None,
                 op2: Op2 | None = None,
                 mu: int = 1, lambda_: int = 1,
                 br: float | None = None,
                 name: str = "ea") -> None:
        """
        Create the Evolutionary Algorithm (EA).

        :param op0: the nullary search operator
        :param op1: the unary search operator
        :param op2: the binary search operator
        :param mu: the number of best solutions to survive in each generation
        :param lambda_: the number of offspring in each generation
        :param br: the rate at which the binary operator is applied
        :param name: the base name of the algorithm
        """
        if op1 is None:
            op1 = Op1()
            if br is None:
                br = 1.0
            elif br != 1.0:
                raise ValueError(
                    f"if op1==None, br must be None or 1.0, but is {br}.")
        elif op1.__class__ is Op1:
            if br is None:
                br = 1.0
            elif br != 1.0:
                raise ValueError(
                    f"if op1 is Op1, br must be None or 1.0, but is {br}.")
        elif (br is not None) and (br == 1.0):
            op1 = Op1()

        if op2 is None:
            op2 = Op2()
            if br is None:
                br = 0.0
            elif br != 0.0:
                raise ValueError(
                    f"if op2==None, br must be None or 0.0, but is {br}.")
        elif op2.__class__ is Op2:
            if br is None:
                br = 0.0
            elif br != 0.0:
                raise ValueError(
                    f"if op2 is Op2, br must be None or 0.0, but is {br}.")
        elif (br is not None) and (br == 0.0):
            op2 = Op2()
        elif mu == 1:
            if br is None:
                br = 0.0
                op2 = Op2()

        if br is None:
            br = 0.2

        name = f"{name}{PART_SEPARATOR}{mu}{PART_SEPARATOR}{lambda_}"
        if 0 < br < 1:
            name = f"{name}{PART_SEPARATOR}{num_to_str_for_name(br)}"
        super().__init__(name, op0, op1, op2)

        if not isinstance(mu, int):
            raise type_error(mu, "mu", int)
        if not (0 < mu <= 1_000_000):
            raise ValueError(f"invalid mu={mu}, must be in 1..1_000_000.")
        if not isinstance(lambda_, int):
            raise type_error(lambda_, "lambda", int)
        if not (0 < lambda_ <= 1_000_000):
            raise ValueError(
                f"invalid lambda={lambda_}, must be in 1..1_000_000.")
        if not isinstance(br, float):
            raise type_error(br, "br", float)
        if not (isfinite(br) and (0.0 <= br <= 1.0)):
            raise ValueError(f"invalid br={br}, must be in [0, 1].")
        if (br > 0.0) and (mu <= 1):
            raise ValueError(
                f"if br (={br}) > 0, then mu (={mu}) must be > 1.")
        #: the number of records to survive in each generation
        self.mu: Final[int] = mu
        #: the number of offsprings per generation
        self.lambda_: Final[int] = lambda_
        #: the rate at which the binary operator is applied
        self.br: Final[float] = br

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of the algorithm to a logger.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        logger.key_value("mu", self.mu)
        logger.key_value("lambda", self.lambda_)
        logger.key_value("br", self.br)
