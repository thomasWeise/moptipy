"""
A fitness-based implementation of a (mu+lambda) Evolutionary Algorithm.

This algorithm is equivalent to :class:`~moptipy.algorithms.so.ea.EA`, but
allows for using a fitness assignment step before selection. This way, we
can implement different diversity enhancement methods such as sharing or
Frequency Fitness Assignment (FFA).
"""
from typing import Final, Union, Callable, List, cast, Optional

from numpy.random import Generator

from moptipy.algorithms.so.ea import EA, _int_0, _float_0
from moptipy.algorithms.so.fitness import Fitness, FRecord, check_fitness
from moptipy.api.operators import Op0, Op1, Op2
from moptipy.api.process import Process
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.strings import PART_SEPARATOR


class FitnessEA(EA):
    """A fitness-based mu+lambda EA."""

    def solve(self, process: Process) -> None:
        """
        Apply the fitness-based EA to an optimization problem.

        :param process: the black-box process object
        """
        mu: Final[int] = self.mu  # mu: number of best solutions kept
        lst_size: Final[int] = mu + self.lambda_  # size = mu + lambda

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
        assign_fitness: Final[Callable[[List[FRecord], Generator], None]] = \
            self.fitness.assign_fitness

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
            lst[i] = FRecord(x, f)  # create and store record

        it: int = 0  # set the iteration counter
        while True:  # lst: keep 0..mu-1, overwrite mu..mu+lambda-1
            it += 1  # step the iteration counter
            for oi in range(mu, lst_size):  # for all lambda offspring
                if should_terminate():      # only continue if we still...
                    return  # have sufficient budget ... otherwise quit
                dest: FRecord = lst[oi]  # pick destination record
                x = dest.x  # the destination "x" value
                dest.it = it  # remember iteration of solution creation

                sx = lst[r0i(mu)].x  # pick a random source record
                if r01() < br:  # apply binary operator at rate br
                    sx2 = sx    # second source "x"
                    while sx2 is sx:     # must be different from sx
                        sx2 = lst[r0i(mu)].x  # get second record
                    op2(random, x, sx, sx2)   # apply binary op
                else:
                    op1(random, x, sx)  # apply unary operator
                dest.f = evaluate(x)  # evaluate new point

            assign_fitness(lst, random)  # assign fitness
            lst.sort()  # best records come first, ties broken by age

    def __init__(self, op0: Op0,
                 op1: Optional[Op1] = None,
                 op2: Optional[Op2] = None,
                 mu: int = 1, lambda_: int = 1,
                 br: Optional[float] = None,
                 fitness: Fitness = None,
                 name: str = "eaf") -> None:
        """
        Create the Evolutionary Algorithm (EA) with fitness assignment.

        :param op0: the nullary search operator
        :param op1: the unary search operator
        :param op2: the binary search operator
        :param mu: the number of best solutions to survive in each generation
        :param lambda_: the number of offspring in each generation
        :param br: the rate at which the binary operator is applied
        :param fitness: the fitness assignment process
        :param name: the base name of the algorithm
        """
        if fitness is None:
            fitness = Fitness()
        if fitness.__class__ is not Fitness:
            name = f"{name}{PART_SEPARATOR}{fitness}"
        super().__init__(op0, op1, op2, mu, lambda_, br, name)
        #: the fitness assignment process
        self.fitness: Final[Fitness] = check_fitness(fitness)

    def log_parameters_to(self, logger: KeyValueLogSection):
        """
        Log the parameters of the algorithm to a logger.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        with logger.scope("v") as v:
            self.fitness.log_parameters_to(v)
