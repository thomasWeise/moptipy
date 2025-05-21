"""
A Hybrid EA-FEA Algorithm: the `EAFEA-D`.

The algorithm has two branches: (1) the EA branch, which performs randomized
local search (RLS), which is in some contexts also called (1+1) EA. (2) the
FEA branch, which performs RLS but uses frequency fitness assignment (FFA)
as optimization criterion. This hybrid algorithm has the following features:

- The new solution of the FEA strand is copied to the EA strand if it is
  accepted by the FEA strand, i.e., if its H-value is <= than that of the
  current solution of that strand.
- The new solution of the EA strand is copied over to the FEA strand if it is
  better than the current solution of the EA strand.
- The H-table is updated by both strands.
- The FEA strand always toggles back to the EA strand.
- The EA strand toggles to the FEA strand if it did not strictly improve for a
  certain a number of FES >= log2 of the total consumed FEs.
"""
from collections import Counter
from math import log2
from typing import Callable, Final

from numpy.random import Generator
from pycommons.types import type_error

from moptipy.algorithms.so.ffa.ffa_h import create_h, log_h
from moptipy.api.algorithm import Algorithm1
from moptipy.api.operators import Op0, Op1
from moptipy.api.process import Process


class EAFEAD(Algorithm1):
    """An implementation of the EAFEA-D."""

    def __init__(self, op0: Op0, op1: Op1, log_h_tbl: bool = False) -> None:
        """
        Create the EAFEA-D.

        :param op0: the nullary search operator
        :param op1: the unary search operator
        :param log_h_tbl: should we log the H table?
        """
        super().__init__("eafeaD", op0, op1)
        if not isinstance(log_h_tbl, bool):
            raise type_error(log_h_tbl, "log_h_tbl", bool)
        #: True if we should log the H table, False otherwise
        self.log_h_tbl: Final[bool] = log_h_tbl

    def solve(self, process: Process) -> None:
        """
        Apply the EAFEA-D to an optimization problem.

        :param process: the black-box process object
        """
        # Create records for old and new point in the search space.
        x_ea = process.create()  # record for current solution of the EA
        x_fea = process.create()  # record for current solution of the FEA
        x_new = process.create()  # record for new solution

        # Obtain the random number generator.
        random: Final[Generator] = process.get_random()

        # Put function references in variables to save time.
        evaluate: Final[Callable] = process.evaluate  # the objective
        should_terminate: Final[Callable] = process.should_terminate
        xcopy: Final[Callable] = process.copy  # copy(dest, source)
        op0: Final[Callable] = self.op0.op0  # the nullary operator
        op1: Final[Callable] = self.op1.op1  # the unary operator

        h, ofs = create_h(process)  # Allocate the h-table

        # Start at a random point in the search space and evaluate it.
        op0(random, x_ea)  # Create 1 solution randomly and
        y_ea: int | float = evaluate(x_ea) + ofs  # evaluate it.
        xcopy(x_fea, x_ea)  # FEA and EA start with the same initial solution.
        y_fea: int | float = y_ea

        ea_no_lt_moves: int = 0  # current no-improvement moves
        fes: int = 1
        use_ffa: bool = False  # We start with the EA branch.
        toggle_ea_to_fea: bool = False  # toggle from ea to fea

        while not should_terminate():  # Until we need to quit...
            # Sample and evaluate new solution.
            op1(random, x_new, x_fea if use_ffa else x_ea)
            y_new: int | float = evaluate(x_new) + ofs
            fes += 1  # and increment the FEs
            h[y_new] += 1  # type: ignore  # Always update H.

            if use_ffa:  # The FEA branch uses FFA.
                use_ffa = False  # Always toggle use from FFA to EA.
                toggle_ea_to_fea = False  # do not immediately toogle back

                h[y_fea] += 1  # type: ignore  # Update H for FEA solution.
                if h[y_new] <= h[y_fea]:  # type: ignore  # FEA acceptance.
                    xcopy(x_ea, x_new)  # Copy solution also to EA.
                    x_fea, x_new = x_new, x_fea
                    y_fea = y_ea = y_new

            else:  # EA or RLS branch performs local search.
                h[y_ea] += 1  # type: ignore  # Update H in *both* branches.

                if y_new <= y_ea:  # The acceptance criterion of RLS / EA.
                    if y_new < y_ea:  # Check if we did an actual improvement.
                        ea_no_lt_moves = 0  # non-improving moves counter = 0.
                        xcopy(x_fea, x_new)  # Copy solution over to FEA.
                        y_fea = y_new  # And store the objective value.
                    else:  # The move was *not* an improvement:
                        ea_no_lt_moves += 1  # Increase non-improved counter.
                        toggle_ea_to_fea = ea_no_lt_moves >= log2(fes)

                    x_ea, x_new = x_new, x_ea  # Accept new solution.
                    y_ea = y_new  # Store objective value.
                else:  # The move was worse than the current solution.
                    ea_no_lt_moves += 1  # Increase non-improvement counter.
                    toggle_ea_to_fea = ea_no_lt_moves >= log2(fes)

                if toggle_ea_to_fea:  # Toggle: EA to FEA.
                    ea_no_lt_moves = 0  # Reset non-improving move counter.
                    use_ffa = True  # Toggle to FFA.

        if not self.log_h_tbl:
            return  # we are done here

        # After we are done, we want to print the H-table.
        if h[y_ea] == 0:  # type: ignore  # Fix the H-table for the case
            h = Counter()   # that only one FE was performed: In this case,
            h[y_ea] = 1  # make Counter with only a single 1 value inside.

        log_h(process, h, ofs)  # log the H-table
