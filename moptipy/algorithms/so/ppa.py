"""A simple implementation of a Plant Propagation Algorithm (PPA)."""
from typing import Callable, Final, cast

from numpy.random import Generator

from moptipy.algorithms.so.record import Record
from moptipy.api.algorithm import Algorithm1
from moptipy.api.operators import Op0, Op1WithStepSize
from moptipy.api.process import Process
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.strings import PART_SEPARATOR
from moptipy.utils.types import check_int_range, type_error


def _int_0(_: int) -> int:
    """
    Return an integer with value `0`.

    :retval 0: always
    """
    return 0


# start book
class PPA(Algorithm1):
    """The Plant Propagation Algorithm (PPA)."""

    def solve(self, process: Process) -> None:
        """
        Apply the PPA to an optimization problem.

        :param process: the black-box process object
        """
        m: Final[int] = self.m  # m: the number of best solutions kept
        nmax: Final[int] = self.nmax  # maximum offspring per solution
        list_len: Final[int] = (nmax + 1) * m
        # initialization of some variables omitted in book for brevity
        # end book
        random: Final[Generator] = process.get_random()  # random gen
        create: Final[Callable] = process.create  # create x container
        evaluate: Final[Callable] = process.evaluate  # the objective
        op0: Final[Callable] = self.op0.op0  # the nullary operator
        op1: Final[Callable] = cast(Op1WithStepSize,
                                    self.op1).op1  # the unary operator
        should_terminate: Final[Callable] = process.should_terminate
        r01: Final[Callable[[], float]] = cast(  # random floats
            Callable[[], float], random.random)
        # start book
        # create list of m random records and enough empty records
        lst: Final[list] = [None] * list_len  # pre-allocate list
        f: int | float = 0  # variable to hold objective values
        for i in range(list_len):  # fill list of size m*nmax
            x = create()  # by creating point in search space
            if i < m:  # only the first m records are initialized by
                op0(random, x)  # applying nullary operator = randomize
                if should_terminate():  # should we quit?
                    return   # computational budget exhausted -> quit
                f = evaluate(x)  # continue? ok, evaluate new solution
            lst[i] = Record(x, f)  # create and store record

        it: int = 0  # the iteration counter
        while True:  # lst: keep 0..mu-1, overwrite mu..mu+lambda-1
            it = it + 1  # step iteration counter
            fmin = frange = lst[0].f  # get range of objective values
            for i in range(m):  # iterate over selected individuals
                fval = lst[i].f  # get objective value
                if fval < fmin:  # is it less than minimum?
                    fmin = fval  # yes -> update the minimum
                elif fval > frange:  # no! is it more than maximum then?
                    frange = fval  # yes -> update maximum
            frange = frange - fmin  # compute the range of objective
            all_same = frange <= 0.0  # do all elements have same f?
            total = m  # the total population length (so far: m)
            for i in range(m):  # generate offspring for each survivor
                rec = lst[i]  # get parent record
                fit = r01() if all_same else ((rec.f - fmin) / frange)
                x = rec.x  # the parent x
                for _ in range(1 + int((1.0 - fit) * r01() * nmax)):
                    if should_terminate():  # should we quit?
                        return  # yes - then return
                    dest = lst[total]  # get next destination record
                    total = total + 1  # remember we have now one more
                    dest.it = it  # set iteration counter
                    op1(random, dest.x, x, fit * r01())  # step-sized op
                    dest.f = evaluate(dest.x)  # evaluate new point
            ls = lst[0:total]
            ls.sort()  # finally, only sort the used elements
            lst[0:total] = ls
# end book

    def __init__(self, op0: Op0, op1: Op1WithStepSize, m: int = 30,
                 nmax: int = 5, name: str = "ppa") -> None:
        """
        Create the Plant Propagation Algorithm (PPA).

        :param op0: the nullary search operator
        :param op1: the unary search operator
        :param m: the number of best solutions to survive in each generation
        :param nmax: the maximum number of offspring per solution
        :param name: the base name of the algorithm
        """
        if not isinstance(op1, Op1WithStepSize):
            raise type_error(op1, "op1", Op1WithStepSize)
        name = f"{name}{PART_SEPARATOR}{m}{PART_SEPARATOR}{nmax}"
        super().__init__(name, op0, op1)

        #: the number of records to survive in each generation
        self.m: Final[int] = check_int_range(m, "m", 1, 1_000_000)
        #: the maximum number of offsprings per solution per iteration
        self.nmax: Final[int] = check_int_range(
            nmax, "nmax", 1, 1_000_000)

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of the algorithm to a logger.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        logger.key_value("m", self.m)
        logger.key_value("nmax", self.nmax)
