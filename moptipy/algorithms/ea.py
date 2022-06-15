"""
The implementation of a (mu+lambda) Evolutionary Algorithm.

This is the basic `mu+lambda`-EA works as follows:

1. Start with a list `lst` of `mu` random records and `lambda` blank records.
2. In each iteration:
    2.1. Use the `mu` first records as input to the search operators to
         generate `lambda` new points in the search space:
         For each new point to be created, the binary operator is applied
         with probability `0<=br<=1` and the unary operator is used otherwise.

         Each of the `mu` records has the same chance to produce such a new
         point in the search space, but no record can be used again (in one
         iteration) until all other `mu-1` selected records have produced at
         least one point. In other words, if `lambda > mu`, then each of the
         `mu` selected record will produce at least `lambda // mu` offspring
         and at most `1 + lambda // mu`.
    2.1. Sort the list `lst` according to the objective value of the record.
         Ties are broken such that younger record are preferred over older
         ones (if they have the same objective value).

If `mu=1`, `lambda=1`, and `br=0`, then this algorithm is exactly equivalent
to the :class:`~moptipy.algorithms.rls.RLS` if the same unary and nullary
operator are used. It is only a bit slower due to the additional overhead of
maintaining a list of records.
"""
from math import isfinite
from typing import Final, Union, Callable, List, cast, Optional

from numpy.random import Generator

from moptipy.algorithms.utils import Record, _int_0, _float_0
from moptipy.api.algorithm import Algorithm2
from moptipy.api.operators import Op0, Op1, Op2
from moptipy.api.process import Process
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.strings import float_to_str
from moptipy.utils.types import type_error


# start nobinary
class EA(Algorithm2):
    """
    The EA is a population-based algorithm using unary and binary operators.

    It starts with a list of `mu` randomly initialized solutions. In each
    step, it retains the `mu` best solutions and generates `lambda` new
    solutions from them using the unary or binary search operator. From the
    joint set of `mu+lambda` solutions, it again selects the best `mu` ones
    for the next iteration. And so on.
    """

    def solve(self, process: Process) -> None:
        """
        Apply the EA to an optimization problem.

        :param process: the black-box process object
        """
        mu: Final[int] = self.__mu  # mu: number of best solutions kept
        lst_size: Final[int] = mu + self.__lambda  # size = mu + lambda
        # Omitted for brevity: store function references in variables
        # end nobinary
        random: Final[Generator] = process.get_random()  # random gen
        create: Final[Callable] = process.create  # create x container
        evaluate: Final[Callable] = process.evaluate  # the objective
        op0: Final[Callable] = self.op0.op0  # the nullary operator
        op1: Final[Callable] = self.op1.op1  # the unary operator
        op2: Final[Callable] = self.op2.op2  # the binary operator
        br: Final[float] = self.__br  # the application are of binary op
        should_terminate: Final[Callable] = process.should_terminate
        r0i: Final[Callable[[int], int]] = cast(   # only if m > 1, we
            Callable[[int], int], random.integers  # need random
            if mu > 1 else _int_0)                 # indices
        r01: Final[Callable[[], float]] = cast(  # only if 0<br<1, we
            Callable[[], float],                 # need random floats
            random.random if 0 < br < 1 else _float_0)
        # start nobinary
        # create list of mu random records and lambda empty records
        lst: Final[List] = [None] * lst_size  # pre-allocate list
        f: Union[int, float] = 0  # variable to hold objective values
        for i in range(lst_size):  # fill list of size mu+lambda
            x = create()  # by creating point in search space
            if i < mu:  # only the first mu records are initialized by
                op0(random, x)  # applying nullary operator = randomize
                if should_terminate():  # should we quit?
                    return   # computational budget exhausted -> quit
                f = evaluate(x)  # continue? ok, evaluate new solution
            lst[i] = Record(x, f)  # create and store record

        iteration: int = 1  # The first "real" iteration has index 1
        while True:  # lst: keep 0..mu-1, overwrite mu..mu+lambda-1
            end: int = mu  # the end index for available old records
            for oi in range(mu, lst_size):  # for all lambda offspring
                if should_terminate():  # only continue if we still...
                    return  # have sufficient budget ... otherwise quit

                offspring: Record = lst[oi]  # pick offspring
                x = offspring.x  # the point in search space we work on
                offspring.it = iteration  # mark as member of new iter.

                pi: int = r0i(end)  # random record in 0..end-1=unused
                rec: Record = lst[pi]  # pick the unused record
                if end > 1:  # swap rec to the end to not use again
                    end = end - 1  # reduce number of unused records
                    lst[end], lst[pi] = rec, lst[end]  # rec <-> end
                else:  # oh: we have used all records
                    end = mu  # then lambda >= mu and we re-use them
                # end nobinary
                # start binary
                if r01() < br:  # apply binary operator at rate br
                    pi = r0i(end)  # random record in 0..end-1
                    rec2: Record = lst[pi]  # pick second unused record
                    if end > 1:  # swap rec2 to end to not use again
                        end = end - 1  # reduce num. of unused records
                        lst[end], lst[pi] = rec2, lst[end]
                    else:  # oh: we have used all records?
                        end = mu  # then lambda>=mu and we re-use them
                    op2(random, x, rec.x, rec2.x)  # apply binary op
                    offspring.f = evaluate(x)  # evaluate new point
                    continue  # below is "else" part with unary operat.
                # end binary
                # start nobinary
                op1(random, x, rec.x)  # apply unary operator to rec.x
                offspring.f = evaluate(x)  # evaluate new point

            iteration = iteration + 1  # step iteration counter
            lst.sort()  # sort list: best records come first
# end nobinary

    def __init__(self, op0: Op0,
                 op1: Optional[Op1] = None,
                 op2: Optional[Op2] = None,
                 mu: int = 1, lambda_: int = 1,
                 br: Optional[float] = None) -> None:
        """
        Create the Evolutionary Algorithm (EA) without binary crossover.

        :param op0: the nullary search operator
        :param op1: the unary search operator
        :param op2: the binary search operator
        :param mu: the number of best solutions to survive in each generation
        :param lambda_: the number of offspring in each generation
        :param br: the rate at which the binary operator is applied
        """
        if op1 is None:
            op1 = Op1()
            if br is None:
                br = 1.0
            elif br != 1.0:
                raise ValueError(
                    f"if op1==None, br must be None or 1.0, but is {br}.")
        elif op1.__class__ == Op1:
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
        elif op2.__class__ == Op2:
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

        super().__init__(
            f"ea_{mu}_{lambda_}_{float_to_str(br).replace('.', 'd')}"
            if 0 < br < 1 else f"ea_{mu}_{lambda_}", op0, op1, op2)

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
        self.__mu: Final[int] = mu
        #: the number of offsprings per generation
        self.__lambda: Final[int] = lambda_
        #: the rate at which the binary operator is applied
        self.__br: Final[float] = br

    def log_parameters_to(self, logger: KeyValueLogSection):
        """
        Log the parameters of the algorithm to a logger.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        logger.key_value("mu", self.__mu)
        logger.key_value("lambda", self.__lambda)
        logger.key_value("br", self.__br)
