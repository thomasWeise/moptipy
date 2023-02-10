"""
The implementation of the hill climbing algorithm with restarts `hcr`.

This algorithm basically works like the normal hill climber `hc`
(:class:`~moptipy.algorithms.so.hill_climber.HillClimber`), but it will
restart automatically if no move was successful for
`max_moves_without_improvement` iterative steps. It therefore maintains an
internal counter `count` which is set to zero at the beginning of each restart
and which is also set to zero again any time a move successfully improved the
best-so-far solution of the current restart. If a search move, i.e., an
application of the unary operator, yielded a new solution which is not better
than the best-so-far solution of the current restart, `count` is incremented.
If `count >= max_moves_without_improvement`, the algorithm begins a new
restart with a new random solution.

1. Thomas Weise. *Optimization Algorithms.* 2021. Hefei, Anhui, China:
   Institute of Applied Optimization (IAO), School of Artificial Intelligence
   and Big Data, Hefei University. http://thomasweise.github.io/oa/
"""
from typing import Callable, Final

from numpy.random import Generator

from moptipy.api.algorithm import Algorithm1
from moptipy.api.operators import Op0, Op1
from moptipy.api.process import Process
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.types import check_int_range


# start book
class HillClimberWithRestarts(Algorithm1):
    """
    The stochastic hill climbing algorithm only accepts improving moves.

    In each step, a hill climber creates a modified copy `new_x` of the
    current best solution `best_x`. If `new_x` is better than `best_x`,
    it becomes the new `best_x`. Otherwise, it is discarded. If no
    improvement is made for `max_moves_without_improvement` steps, the
    algorithm restarts.
    """

    def solve(self, process: Process) -> None:
        """
        Apply the hill climber with restarts to an optimization problem.

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
        limit: Final[int] = self.max_moves_without_improvement

        while not should_terminate():  # Until we need to quit....
            self.op0.op0(random, best_x)  # Create random solution and
            best_f: int | float = evaluate(best_x)
            count: int = 0  # The counter of unsuccessful moves = 0.

            while not should_terminate():  # Until we need to quit...
                op1(random, new_x, best_x)  # new_x=neighbor of best_x
                new_f: int | float = evaluate(new_x)
                if new_f < best_f:  # new_x is _better_ than best_x?
                    best_f = new_f  # Store its objective value.
                    best_x, new_x = new_x, best_x  # Swap best and new.
                    count = 0  # Reset unsuccessful move counter.
                else:  # The move did not lead to an improvement!?
                    count += 1  # Increment unsuccessful move counter.
                    if count >= limit:  # Too many unsuccessful moves?
                        break  # Break inner loop, start again randomly.
# end book

    def __init__(self, op0: Op0, op1: Op1,
                 max_moves_without_improvement: int) -> None:
        """
        Create the hill climber.

        :param op0: the nullary search operator
        :param op1: the unary search operator
        :param max_moves_without_improvement: the maximum number of
            moves without improvement before a restart
        """
        super().__init__(
            f"hcr_{max_moves_without_improvement}", op0, op1)
        #: the maximum moves without improvement
        self.max_moves_without_improvement: Final[int] = \
            check_int_range(
                max_moves_without_improvement,
                "max_moves_without_improvement", 1, 1_000_000_000_000)

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log all parameters of this component as key-value pairs.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        logger.key_value("maxMovesWithoutImprovement",
                         self.max_moves_without_improvement)
