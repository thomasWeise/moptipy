"""
The EAFEA is hybrid of the (1+1)FEA and the (1+1)EA without Solution Transfer.

The algorithm has two branches: (1) the EA branch, which performs randomized
local search (RLS), which is in some contexts also called (1+1) EA. (2) the
FEA branch, which performs RLS but uses frequency fitness assignment (FFA)
as optimization criterion. No flow of information takes place between the two
branches.

1. Thomas Weise, Zhize Wu, Xinlu Li, Yan Chen, and Jörg Lässig. Frequency
   Fitness Assignment: Optimization without Bias for Good Solutions can be
   Efficient. *IEEE Transactions on Evolutionary Computation (TEVC)*.
   27(4):980-992. August 2023.
   doi: https://doi.org/10.1109/TEVC.2022.3191698
"""
from collections import Counter
from typing import Callable, Final

from numpy.random import Generator
from pycommons.types import type_error

from moptipy.algorithms.so.ffa.ffa_h import create_h, log_h
from moptipy.api.algorithm import Algorithm1
from moptipy.api.operators import Op0, Op1
from moptipy.api.process import Process


class EAFEA(Algorithm1):
    """An implementation of the EAFEA."""

    def __init__(self, op0: Op0, op1: Op1, log_h_tbl: bool = False) -> None:
        """
        Create the EAFEA.

        :param op0: the nullary search operator
        :param op1: the unary search operator
        :param log_h_tbl: should we log the H table?
        """
        super().__init__("eafea", op0, op1)
        if not isinstance(log_h_tbl, bool):
            raise type_error(log_h_tbl, "log_h_tbl", bool)
        #: True if we should log the H table, False otherwise
        self.__log_h_tbl: Final[bool] = log_h_tbl

    def solve(self, process: Process) -> None:
        """
        Apply the EAFEA to an optimization problem.

        :param process: the black-box process object
        """
        # Create records for old and new point in the search space.
        x_c = process.create()  # record for current solution of the EA
        x_d = process.create()  # record for current solution of the FEA
        x_n = process.create()  # record for new solution

        # Obtain the random number generator.
        random: Final[Generator] = process.get_random()

        # Put function references in variables to save time.
        evaluate: Final[Callable] = process.evaluate  # the objective
        should_terminate: Final[Callable] = process.should_terminate
        op0: Final[Callable] = self.op0.op0  # the nullary operator
        op1: Final[Callable] = self.op1.op1  # the unary operator

        h, ofs = create_h(process)  # Allocate the h-table

        # Start at a random point in the search space and evaluate it.
        op0(random, x_c)  # Create 1 solution randomly and
        y_c: int | float = evaluate(x_c) + ofs  # evaluate it.
        process.copy(x_d, x_c)
        y_d: int | float = y_c
        use_ffa: bool = True

        while not should_terminate():  # Until we need to quit...
            use_ffa = not use_ffa  # toggle use of FFA
            op1(random, x_n, x_d if use_ffa else x_c)
            y_n: int | float = evaluate(x_n) + ofs

            if use_ffa:  # the FEA branch
                h[y_n] += 1  # type: ignore  # Increase the frequency
                h[y_d] += 1  # type: ignore  # of new_f and cur_f.
                if h[y_n] <= h[y_d]:  # type: ignore
                    y_d = y_n  # Store its objective value.
                    x_d, x_n = x_n, x_d  # Swap best and new.
            elif y_n <= y_c:  # the EA/RLS branch
                y_c = y_n
                x_c, x_n = x_n, x_c

        if not self.__log_h_tbl:
            return  # we are done here

        # After we are done, we want to print the H-table.
        if h[y_c] == 0:  # type: ignore  # Fix the H-table for the case
            h = Counter()   # that only one FE was performed: In this case,
            h[y_c] = 1  # make Counter with only a single 1 value inside.

        log_h(process, h, ofs)  # log the H-table
