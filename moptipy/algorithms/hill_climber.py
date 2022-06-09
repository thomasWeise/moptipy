"""
The implementation of the basic hill climbing algorithm `hc`.

The algorithm starts by applying the nullary search operator, an
implementation of :meth:`~moptipy.api.operators.Op0.op0`, to sample one fully
random solution. This is the first best-so-far solution. In each step, it
applies the unary operator, an implementation of
:meth:`~moptipy.api.operators.Op1.op1`, to the best-so-far solution to obtain
a new, similar solution. If this new solution is strictly better than the
current best-so-far solution, it replaces this solution. Otherwise, it is
discarded.

The hill climbing algorithm is a simple local search that only accepts
strictly improving moves. It is thus similar to the randomized local search
(`rls`) implemented in :class:`~moptipy.algorithms.rls.RLS`, which, however,
accepts non-deteriorating moves. We also provide `hcr`, a variant of the hill
climber that restarts automatically with a certain number of moves were not
able to improve the current best-so-far solution in class :class:`~moptipy.\
algorithms.hill_climber_with_restarts.HillClimberWithRestarts`.
"""
from typing import Final, Union, Callable

from numpy.random import Generator

from moptipy.api.algorithm import Algorithm1
from moptipy.api.operators import Op0, Op1
from moptipy.api.process import Process
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.types import type_error


# start book
class HillClimber(Algorithm1):
    """
    The stochastic hill climbing algorithm only accepts improving moves.

    In each step, a hill climber creates a modified copy `new_x` of the
    current best solution `best_x`. If `new_x` is better than `best_x`,
    it becomes the new `best_x`. Otherwise, it is discarded.
    """

    def solve(self, process: Process) -> None:
        """
        Apply the hill climber to an optimization problem.

        :param process: the black-box process object
        """
        # Create records for old and new point in the search space.
        best_x = process.create()  # record for best-so-far solution
        new_x = process.create()  # record for new solution
        # Obtain the random number generator.
        random: Final[Generator] = process.get_random()

        # Put function references in variables to save time.
        evaluate: Final[Callable] = process.evaluate  # the objective
        op1: Final[Callable] = self.op1.op1  # the unary operator
        should_terminate: Final[Callable] = process.should_terminate

        # Start at a random point in the search space and evaluate it.
        self.op0.op0(random, best_x)  # Create 1 solution randomly and
        best_f: Union[int, float] = evaluate(best_x)  # evaluate it.

        while not should_terminate():  # Until we need to quit...
            op1(random, new_x, best_x)  # new_x = neighbor of best_x
            new_f: Union[int, float] = evaluate(new_x)
            if new_f < best_f:  # new_x is _better_ than best_x?
                best_f = new_f  # Store its objective value.
                best_x, new_x = new_x, best_x  # Swap best and new.
# end book

    def __solve_seeded(self, process: Process) -> None:
        """
        Apply the hill climber with a seed to an optimization problem.

        :param process: the black-box process object
        """
        # Create records for old and new point in the search space.
        best_x = process.create()  # record for best-so-far solution
        new_x = process.create()  # record for new solution
        # Obtain the random number generator.
        random: Final[Generator] = process.get_random()

        # Put function references in variables to save time.
        evaluate: Final[Callable] = process.evaluate  # the objective
        op1: Final[Callable] = self.op1.op1  # the unary operator
        should_terminate: Final[Callable] = process.should_terminate

        # Start at an existing point in the search space and get its quality.
        process.get_copy_of_best_x(best_x)  # get the best-so-far solution
        best_f: Union[int, float] = process.get_best_f()  # get the quality.

        while not should_terminate():  # Until we need to quit...
            op1(random, new_x, best_x)  # new_x = neighbor of best_x
            new_f: Union[int, float] = evaluate(new_x)
            if new_f < best_f:  # new_x is _better_ than best_x?
                best_f = new_f  # Store its objective value.
                best_x, new_x = new_x, best_x  # Swap best and new.

    def __init__(self, op0: Op0, op1: Op1,
                 seeded: bool = False) -> None:
        """
        Create the hill climber.

        :param op0: the nullary search operator
        :param op1: the unary search operator
        :param seeded: `True` if the algorithm should be run in a seeded
            fashion, i.e., expect an existing best solution. `False` if
            it should run in the traditional way, starting at a random
            solution
        """
        super().__init__("hc", op0, op1)
        if not isinstance(seeded, bool):
            raise type_error(seeded, "seeded", bool)
        if seeded:
            self.solve = self.__solve_seeded  # type: ignore
        #: was this algorithm started in its seeded fashion?
        self.__seeded: Final[bool] = seeded

    def log_parameters_to(self, logger: KeyValueLogSection):
        """
        Log the parameters of the algorithm to a logger.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        logger.key_value("seeded", self.__seeded)
