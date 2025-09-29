"""
A Hybrid SA-FEA Algorithm: the `SAFEA-N`.

This hybrid algorithm has the following features:

- The new solution of the FEA strand is always copied to the SA strand.
- The new solution of the SA strand is copied over to the FEA strand if it is
  better than the current solution of the FEA strand.
- The H-table is updated by both strands.
- The FEA strand always toggles back to the SA strand.
- The SA strand toggles to the FEA strand if it did not improve for a certain,
  increasing `time limit`.
- Every time the SA strand toggles over to the FEA strand, `time_limit` is
  incremented by 1.
"""
from collections import Counter
from math import exp
from typing import Callable, Final

from numpy.random import Generator
from pycommons.types import type_error

from moptipy.algorithms.modules.temperature_schedule import TemperatureSchedule
from moptipy.algorithms.so.ffa.ffa_h import create_h, log_h
from moptipy.api.algorithm import Algorithm1
from moptipy.api.operators import Op0, Op1
from moptipy.api.process import Process
from moptipy.utils.logger import KeyValueLogSection


class SAFEAN(Algorithm1):
    """An implementation of the SAFEA-N."""

    def __init__(self, op0: Op0, op1: Op1, schedule: TemperatureSchedule,
                 log_h_tbl: bool = False) -> None:
        """
        Create the SAFEA-N.

        :param op0: the nullary search operator
        :param op1: the unary search operator
        :param schedule: the temperature schedule to use
        :param log_h_tbl: should we log the H table?
        """
        if not isinstance(schedule, TemperatureSchedule):
            raise type_error(schedule, "schedule", TemperatureSchedule)
        if not isinstance(log_h_tbl, bool):
            raise type_error(log_h_tbl, "log_h_tbl", bool)
        super().__init__(f"safeaN_{schedule}", op0, op1)
        #: True if we should log the H table, False otherwise
        self.__log_h_tbl: Final[bool] = log_h_tbl
        #: the temperature schedule
        self.schedule: Final[TemperatureSchedule] = schedule

    def solve(self, process: Process) -> None:
        """
        Apply the SAFEA-N to an optimization problem.

        :param process: the black-box process object
        """
        # Create records for old and new point in the search space.
        x_ea = process.create()  # record for current solution of the SA
        x_fea = process.create()  # record for current solution of the FEA
        x_new = process.create()  # record for new solution

        # Obtain the random number generator.
        random: Final[Generator] = process.get_random()

        # Put function references in variables to save time.
        evaluate: Final[Callable] = process.evaluate  # the objective
        should_terminate: Final[Callable] = process.should_terminate
        temperature: Final[Callable[[int], float]] = self.schedule.temperature
        r01: Final[Callable[[], float]] = random.random  # random from [0, 1)
        xcopy: Final[Callable] = process.copy  # copy(dest, source)
        op0: Final[Callable] = self.op0.op0  # the nullary operator
        op1: Final[Callable] = self.op1.op1  # the unary operator

        h, ofs = create_h(process)  # Allocate the h-table

        # Start at a random point in the search space and evaluate it.
        op0(random, x_ea)  # Create 1 solution randomly and
        y_ea: int | float = evaluate(x_ea) + ofs  # evaluate it.
        xcopy(x_fea, x_ea)  # FEA and SA start with the same initial solution.
        y_fea: int | float = y_ea

        sa_max_no_lt_moves: int = 1  # maximum no-improvement moves for SA
        sa_no_lt_moves: int = 0  # current no-improvement moves
        use_ffa: bool = False  # We start with the SA branch.
        tau: int = 0  # The iteration index, needs to be 0 at first cmp.

        while not should_terminate():  # Until we need to quit...
            # Sample and evaluate new solution.
            op1(random, x_new, x_fea if use_ffa else x_ea)
            y_new: int | float = evaluate(x_new) + ofs
            h[y_new] += 1  # type: ignore  # Always update H.

            if use_ffa:  # The FEA branch uses FFA.
                use_ffa = False  # Always toggle use from FFA to SA.
                sa_no_lt_moves = 0  # Reset the SA no-improv move counter.

                h[y_fea] += 1  # type: ignore  # Update H for FEA solution.
                if h[y_new] <= h[y_fea]:  # type: ignore  # FEA acceptance.
                    xcopy(x_ea, x_new)  # Copy solution also to SA.
                    x_fea, x_new = x_new, x_fea
                    y_fea = y_new
                else:  # FEA does not accept, but we always copy to the SA,
                    x_ea, x_new = x_new, x_ea  # so we quickly swap here.
                y_ea = y_new  # we always copy the solution over to the SA

            else:  # SA branch performs simulated annealing.
                h[y_ea] += 1  # type: ignore  # Update H in *both* branches.

                if y_new < y_fea:  # Is new solution better than the FEA one?
                    xcopy(x_fea, x_new)  # Copy solution over to FEA.
                    y_fea = y_new  # And store the objective value.

                if (y_new <= y_ea) or (  # Accept if <= or if SA criterion
                        r01() < exp((y_ea - y_new) / temperature(tau))):
                    x_ea, x_new = x_new, x_ea  # Accept new solution.
                    y_new, y_ea = y_ea, y_new  # Swap values (for line below).
                    if y_new > y_ea:  # Check if we did an actual improvement.
                        sa_no_lt_moves = 0  # non-improving moves counter = 0.
                        continue  # We can jump to the next iteration.

                sa_no_lt_moves += 1  # Increase non-improved counter.
                if sa_no_lt_moves >= sa_max_no_lt_moves:  # Toggle: SA to FEA.
                    sa_max_no_lt_moves += 1  # Increment limit by one.
                    use_ffa = True  # Toggle to FFA.
            tau += 1  # Step the iteration index.

        if not self.__log_h_tbl:
            return  # we are done here

        # After we are done, we want to print the H-table.
        if h[y_ea] == 0:  # type: ignore  # Fix the H-table for the case
            h = Counter()   # that only one FE was performed: In this case,
            h[y_ea] = 1  # make Counter with only a single 1 value inside.

        log_h(process, h, ofs)  # log the H-table

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log all parameters of the SAFEA-N algorithm.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        with logger.scope("ts") as ts:
            self.schedule.log_parameters_to(ts)
