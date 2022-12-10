"""A simple implementation of a (mu+lambda) Memetic Algorithm."""
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
from moptipy.utils.types import type_error


# start book
class MA(Algorithm0):
    """An MA is a population-based algorithm using binary operators."""

    def solve(self, process: Process) -> None:
        """
        Apply the MA to an optimization problem.

        :param process: the black-box process object
        """
        # Omitted for brevity: store function references in variables
        # end book
        mu: Final[int] = self.mu  # mu: number of best solutions kept
        mu_plus_lambda: Final[int] = mu + self.lambda_  # size
        random: Final[Generator] = process.get_random()  # random gen
        create: Final[Callable] = process.create  # create x container
        evaluate: Final[Callable] = process.evaluate  # the objective
        op0: Final[Callable] = self.op0.op0  # the nullary operator
        op2: Final[Callable] = self.op2.op2  # the binary operator
        ls_fes: Final[int] = self.ls_fes  # the number of FEs per ls run
        ls_solve: Final[Callable[[Process], None]] = self.ls.solve
        forward_ls_op_to: Final[Callable] = cast(  # forward starting
            Op0Forward, self.ls.op0).forward_to  # point of ls to...
        should_terminate: Final[Callable] = process.should_terminate
        r0i: Final[Callable[[int], int]] = cast(  # random integer
            Callable[[int], int], random.integers)
        # start book
        # create list of mu random records and lambda empty records
        lst: Final[list] = [None] * mu_plus_lambda  # pre-allocate list
        f: int | float = 0  # variable to hold objective values
        for i in range(mu_plus_lambda):  # fill list of size mu+lambda
            x = create()  # by creating point in search space
            if i < mu:  # only the first mu records are initialized by
                op0(random, x)  # applying nullary operator = randomize
                if should_terminate():  # should we quit?
                    return   # computational budget exhausted -> quit
                with for_fes(process, ls_fes) as s1:  # fe-limited proc
                    with from_starting_point(s1, x, evaluate(x)) as s2:
                        forward_ls_op_to(s2.get_copy_of_best_x)
                        ls_solve(s2)  # apply local search modifying x
                        f = s2.get_best_f()  # get quality of x
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
                sx2 = sx    # second source "x"
                while sx2 is sx:     # must be different from sx
                    sx2 = lst[r0i(mu)].x  # get second record
                op2(random, x, sx, sx2)  # apply binary op

                with for_fes(process, ls_fes) as s1:  # fe-limited proc
                    with from_starting_point(s1, x, evaluate(x)) as s2:
                        forward_ls_op_to(s2.get_copy_of_best_x)
                        ls_solve(s2)  # apply local search modifying x
                        dest.f = s2.get_best_f()  # get quality of x
            lst.sort()  # best records come first, ties broken by age
            # end book
            cast(Op0Forward, self.ls.op0).stop_forwarding()  # clean up

    def __init__(self, op0: Op0,
                 op2: Op2, ls: Algorithm0,
                 mu: int = 32, lambda_: int = 32,
                 ls_fes: int = 1000,
                 name: str = "ma") -> None:
        """
        Create the Evolutionary Algorithm (EA).

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
        if not isinstance(mu, int):
            raise type_error(mu, "mu", int)
        if not (1 < mu <= 1_000_000):
            raise ValueError(f"invalid mu={mu}, must be in 2..1_000_000.")
        if not isinstance(lambda_, int):
            raise type_error(lambda_, "lambda", int)
        if not (0 < lambda_ <= 1_000_000):
            raise ValueError(
                f"invalid lambda={lambda_}, must be in 1..1_000_000.")
        if not isinstance(ls_fes, int):
            raise type_error(ls_fes, "ls_fes", int)
        if not (0 < ls_fes <= 100_000_000):
            raise ValueError(
                f"invalid ls_fes={ls_fes}, must be in 1..100_000_000.")
        if not isinstance(ls, Algorithm0):
            raise type_error(ls, "ls", Algorithm0)
        if not isinstance(ls.op0, Op0Forward):
            raise type_error(ls.op0, "ls.op0", Op0Forward)

        #: the number of records to survive in each generation
        self.mu: Final[int] = mu
        #: the number of offsprings per generation
        self.lambda_: Final[int] = lambda_
        #: the number of FEs per local search run
        self.ls_fes: Final[int] = ls_fes
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
        logger.key_value("lsSteps", self.ls_fes)
        with logger.scope(SCOPE_OP2) as o:
            self.op2.log_parameters_to(o)
        with logger.scope("ls") as ls:
            self.ls.log_parameters_to(ls)

    def initialize(self) -> None:
        """Initialize the memetic algorithm."""
        super().initialize()
        self.ls.initialize()
        self.op2.initialize()
