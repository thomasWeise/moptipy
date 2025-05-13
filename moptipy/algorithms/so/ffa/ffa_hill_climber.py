"""
The FFA-based version of the Hill Climber: The FHC.

This algorithm is based on
:class:`~moptipy.algorithms.so.hill_climber.HillClimber`, i.e., the simple
Hill Climber, but uses Frequency Fitness Assignment (FFA) as fitness
assignment process. FFA replaces all objective values with their encounter
frequencies in the selection decisions. The more often an objective value is
encountered, the higher gets its encounter frequency. Therefore, local
optima are slowly receiving worse and worse fitness.

This algorithm is closely related
:class:`~moptipy.algorithms.so.ffa.fea1plus1.FEA1plus1`.
The difference is that it accepts new solutions only if their frequency
fitness is *lower* than the frequency fitness of the current solution.
:class:`~moptipy.algorithms.so.ffa.fea1plus1.FEA1plus1`, on the other hand,
accepts solutions if their frequency fitness is *lower or equal*
than the frequency fitness of the current solution.

1. Thomas Weise, Zhize Wu, Xinlu Li, and Yan Chen. Frequency Fitness
   Assignment: Making Optimization Algorithms Invariant under Bijective
   Transformations of the Objective Function Value. *IEEE Transactions on
   Evolutionary Computation* 25(2):307-319. April 2021. Preprint available at
   arXiv:2001.01416v5 [cs.NE] 15 Oct 2020.
   https://dx.doi.org/10.1109/TEVC.2020.3032090
2. Thomas Weise, Zhize Wu, Xinlu Li, Yan Chen, and Jörg Lässig. Frequency
   Fitness Assignment: Optimization without Bias for Good Solutions can be
   Efficient. *IEEE Transactions on Evolutionary Computation (TEVC)*.
   27(4):980-992. August 2023.
   doi: https://doi.org/10.1109/TEVC.2022.3191698
3. Thomas Weise, Mingxu Wan, Ke Tang, Pu Wang, Alexandre Devert, and Xin
   Yao. Frequency Fitness Assignment. *IEEE Transactions on Evolutionary
   Computation (IEEE-EC),* 18(2):226-243, April 2014.
   https://dx.doi.org/10.1109/TEVC.2013.2251885
4. Thomas Weise, Yan Chen, Xinlu Li, and Zhize Wu. Selecting a diverse set of
   benchmark instances from a tunable model problem for black-box discrete
   optimization algorithms. *Applied Soft Computing Journal (ASOC),*
   92:106269, June 2020. https://dx.doi.org/10.1016/j.asoc.2020.106269
5. Thomas Weise, Xinlu Li, Yan Chen, and Zhize Wu. Solving Job Shop Scheduling
   Problems Without Using a Bias for Good Solutions. In *Genetic and
   Evolutionary Computation Conference Companion (GECCO'21 Companion),*
   July 10-14, 2021, Lille, France. ACM, New York, NY, USA.
   ISBN 978-1-4503-8351-6. https://dx.doi.org/10.1145/3449726.3463124
6. Thomas Weise, Mingxu Wan, Ke Tang, and Xin Yao. Evolving Exact Integer
   Algorithms with Genetic Programming. In *Proceedings of the IEEE Congress
   on Evolutionary Computation (CEC'14), Proceedings of the 2014 World
   Congress on Computational Intelligence (WCCI'14),* pages 1816-1823,
   July 6-11, 2014, Beijing, China. Los Alamitos, CA, USA: IEEE Computer
   Society Press. ISBN: 978-1-4799-1488-3.
   https://dx.doi.org/10.1109/CEC.2014.6900292
7. Tianyu Liang, Zhize Wu, Jörg Lässig, Daan van den Berg, Sarah Louise
   Thomson, and Thomas Weise. Addressing the Traveling Salesperson Problem
   with Frequency Fitness Assignment and Hybrid Algorithms. *Soft Computing.*
   2024. https://dx.doi.org/10.1007/s00500-024-09718-8
"""
from collections import Counter
from typing import Callable, Final

from numpy.random import Generator
from pycommons.types import type_error

from moptipy.algorithms.so.ffa.ffa_h import create_h, log_h
from moptipy.api.algorithm import Algorithm1
from moptipy.api.operators import Op0, Op1
from moptipy.api.process import Process


class FHC(Algorithm1):
    """
    The FFA-based version of the Hill Climber.

    This algorithm applies Frequency Fitness Assignment (FFA).
    This means that it does not select solutions based on whether
    they are better or worse. Instead, it selects the solution whose
    objective value has been encountered during the search less often.
    The word "best" therefore is not used in the traditional sense, i.e.,
    that one solution is better than another one terms of its objective
    value. Instead, the current best solution is always the one whose
    objective value we have seen the least often.

    This algorithm implementation works best if objective values are
    integers and have lower and upper bounds that are not too far
    away from each other. If there are too many possible different objective
    values, the algorithm degenerates to a random walk.
    """

    def __init__(self, op0: Op0, op1: Op1, log_h_tbl: bool = False) -> None:
        """
        Create the FHC.

        :param op0: the nullary search operator
        :param op1: the unary search operator
        :param log_h_tbl: should we log the H table?
        """
        super().__init__("fhc", op0, op1)
        if not isinstance(log_h_tbl, bool):
            raise type_error(log_h_tbl, "log_h_tbl", bool)
        #: True if we should log the H table, False otherwise
        self.__log_h_tbl: Final[bool] = log_h_tbl

    def solve(self, process: Process) -> None:
        """
        Apply the FHC to an optimization problem.

        :param process: the black-box process object
        """
        # Create records for old and new point in the search space.
        cur_x = process.create()  # record for current solution
        new_x = process.create()  # record for new solution

        # Obtain the random number generator.
        random: Final[Generator] = process.get_random()

        # Put function references in variables to save time.
        evaluate: Final[Callable] = process.evaluate  # the objective
        should_terminate: Final[Callable] = process.should_terminate
        op0: Final[Callable] = self.op0.op0  # the nullary operator
        op1: Final[Callable] = self.op1.op1  # the unary operator

        h, ofs = create_h(process)  # Allocate the h-table

        # Start at a random point in the search space and evaluate it.
        op0(random, cur_x)  # Create 1 solution randomly and
        cur_f: int | float = evaluate(cur_x) + ofs  # evaluate it.

        while not should_terminate():  # Until we need to quit...
            op1(random, new_x, cur_x)  # new_x = neighbor of cur_x
            new_f: int | float = evaluate(new_x) + ofs

            h[new_f] += 1  # type: ignore  # Increase the frequency
            h[cur_f] += 1  # type: ignore  # of new_f and cur_f.
            if h[new_f] < h[cur_f]:  # type: ignore
                cur_f = new_f  # Store its objective value.
                cur_x, new_x = new_x, cur_x  # Swap best and new.

        if not self.__log_h_tbl:
            return  # we are done here

        # After we are done, we want to print the H-table.
        if h[cur_f] == 0:  # type: ignore  # Fix the H-table for the case
            h = Counter()   # that only one FE was performed: In this case,
            h[cur_f] = 1  # make Counter with only a single 1 value inside.

        log_h(process, h, ofs)  # log the H-table
