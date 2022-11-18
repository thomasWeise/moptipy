"""
The FFA-based version of the (1+1)-EA: the (1+1)-FEA.

This algorithm is based on :class:`~moptipy.algorithms.so.rls.RLS`, i.e., the
(1+1)-EA, but uses Frequency Fitness Assignment (FFA) as fitness assignment
process. FFA replaces all objective values with their encounter frequencies in
the selection decisions. The more often an objective value is encountered, the
higher gets its encounter frequency. Therefore, local optima are slowly
receiving worse and worse fitness.

1. Thomas Weise, Zhize Wu, Xinlu Li, and Yan Chen. Frequency Fitness
   Assignment: Making Optimization Algorithms Invariant under Bijective
   Transformations of the Objective Function Value. *IEEE Transactions on
   Evolutionary Computation* 25(2):307-319. April 2021. Preprint available at
   arXiv:2001.01416v5 [cs.NE] 15 Oct 2020.
   https://dx.doi.org/10.1109/TEVC.2020.3032090
2. Thomas Weise, Zhize Wu, Xinlu Li, Yan Chen, and Jörg Lässig. Frequency
   Fitness Assignment: Optimization without Bias for Good Solutions can be
   Efficient. *IEEE Transactions on Evolutionary Computation (TEVC)*. 2022.
   Early Access. https://dx.doi.org/10.1109/TEVC.2022.3191698
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
"""
from typing import Callable, Final, cast

import numpy as np
from numpy.random import Generator

from moptipy.api.algorithm import Algorithm1
from moptipy.api.operators import Op0, Op1
from moptipy.api.process import Process


class FEA1plus1(Algorithm1):
    """
    The FFA-based version of the (1+1)-EA: the (1+1)-FEA.

    This algorithm applies Frequency Fitness Assignment (FFA).
    This means that it does not select solutions based on whether
    they are better or worse. Instead, it selects the solution whose
    objective value has been encountered during the search less often.
    The word "best" therefore is not used in the traditional sense, i.e.,
    that one solution is better than another one terms of its objective
    value. Instead, the current best solution is always the one whose
    objective value we have seen the least often.

    In each step, a (1+1)-FEA creates a modified copy `new_x` of the
    current best solution `best_x`. It then increments the frequency fitness
    of both solutions by 1. If frequency fitness of `new_x` is not bigger
    the one of `best_x`, it becomes the new `best_x`.
    Otherwise, it is discarded.

    This algorithm implementation requires that objective values are
    integers and have lower and upper bounds that are not too far
    away from each other.
    """

    def __init__(self, op0: Op0, op1: Op1) -> None:
        """
        Create the (1+1)-FEA.

        :param op0: the nullary search operator
        :param op1: the unary search operator
        """
        super().__init__("fea1p1", op0, op1)

    def solve(self, process: Process) -> None:
        """
        Apply the (1+1)-FEA to an optimization problem.

        :param process: the black-box process object
        """
        # Create records for old and new point in the search space.
        best_x = process.create()  # record for best-so-far solution
        new_x = process.create()  # record for new solution
        lb: Final[int] = cast(int, process.lower_bound())

        # h holds the encounter frequency of each objective value.
        h: Final[np.ndarray] = np.zeros(
            cast(int, process.upper_bound()) - lb + 1, np.uint64)
        # Obtain the random number generator.
        random: Final[Generator] = process.get_random()

        # Put function references in variables to save time.
        evaluate: Final[Callable] = process.evaluate  # the objective
        op1: Final[Callable] = self.op1.op1  # the unary operator
        should_terminate: Final[Callable] = process.should_terminate

        # Start at a random point in the search space and evaluate it.
        self.op0.op0(random, best_x)  # Create 1 solution randomly and
        best_f: int = cast(int, evaluate(best_x)) - lb  # evaluate it.

        while not should_terminate():  # Until we need to quit...
            op1(random, new_x, best_x)  # new_x = neighbor of best_x
            new_f: int = cast(int, evaluate(new_x)) - lb

            h[new_f] = h[new_f] + 1  # Increase frequency of new_f and
            h[best_f] = best_h = h[best_f] + 1  # of best_f.
            if h[new_f] <= best_h:  # new_x is no worse than best_x?
                best_f = new_f  # Store its objective value.
                best_x, new_x = new_x, best_x  # Swap best and new.
