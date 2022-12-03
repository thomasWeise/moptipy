"""
The simulated annealing algorithm with configurable temperature schedule.

This algorithm is simular to :class:`~moptipy.algorithms.so.rls.RLS`, but it
sometimes also accepts solutions that are worse than the current one.
"""
from math import exp
from typing import Callable, Final

from numpy.random import Generator

from moptipy.algorithms.so.temperature_schedule import TemperatureSchedule
from moptipy.api.algorithm import Algorithm1
from moptipy.api.operators import Op0, Op1
from moptipy.api.process import Process
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.types import type_error


# start book
class SimulatedAnnealing(Algorithm1):
    """Simulated Annealing is an RLS sometimes accepting worsening moves."""

    def solve(self, process: Process) -> None:
        """
        Apply the simulated annealing algorithm to an optimization problem.

        :param process: the black-box process object
        """
        # Create records for old and new point in the search space.
        best_x = process.create()  # record for best-so-far solution
        new_x = process.create()  # record for new solution
        # Obtain the random number generator.
        random: Final[Generator] = process.get_random()

        # Put function references in variables to for faster calls.
        temparature: Final[Callable[[int], float]] = self.schedule.temperature
        r01: Final[Callable[[], float]] = random.random  # random from [0, 1)
        evaluate: Final[Callable] = process.evaluate  # the objective
        op1: Final[Callable] = self.op1.op1  # the unary operator
        should_terminate: Final[Callable] = process.should_terminate

        # Start at a random point in the search space and evaluate it.
        self.op0.op0(random, best_x)  # Create 1 solution randomly and
        best_f: int | float = evaluate(best_x)  # evaluate it.
        tau: int = 0  # The iteration index, needs to be 0 at first cmp.

        while not should_terminate():  # Until we need to quit...
            op1(random, new_x, best_x)  # new_x = neighbor of best_x
            new_f: int | float = evaluate(new_x)
            if (new_f <= best_f) or (  # Accept if <= or if SA criterion
                    r01() < exp((new_f - best_f) / temparature(tau))):
                best_f = new_f  # Store its objective value.
                best_x, new_x = new_x, best_x  # Swap best and new.
            tau = tau + 1  # Step the iteration index.
# end book

    def __init__(self, op0: Op0, op1: Op1,
                 schedule: TemperatureSchedule) -> None:
        """
        Create the simulated annealing algorithm.

        :param op0: the nullary search operator
        :param op1: the unary search operator
        :param schedule: the temperature schedule to use
        """
        super().__init__(f"sa_{schedule}", op0, op1)
        if not isinstance(schedule, TemperatureSchedule):
            raise type_error(schedule, "schedule", TemperatureSchedule)
        #: the temperature schedule
        self.schedule: Final[TemperatureSchedule] = schedule

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log all parameters of the simulated annealing algorithm.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        with logger.scope("ts") as ts:
            self.schedule.log_parameters_to(ts)
