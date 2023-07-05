"""
The FFA-based version of the (1+1)-EA: the (1+1)-FEA.

This algorithm is based on :class:`~moptipy.algorithms.so.rls.RLS`, i.e., the
(1+1)-EA, but uses Frequency Fitness Assignment (FFA) as fitness assignment
process. FFA replaces all objective values with their encounter frequencies in
the selection decisions. The more often an objective value is encountered, the
higher gets its encounter frequency. Therefore, local optima are slowly
receiving worse and worse fitness.

FFA is also implemented as a fitness assignment process
(:mod:`~moptipy.algorithms.so.fitness`) in module
:mod:`~moptipy.algorithms.so.fitnesses.ffa`.

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
from io import StringIO
from typing import Callable, Final, Iterable, cast

import numpy as np
from numpy.random import Generator

from moptipy.api.algorithm import Algorithm1
from moptipy.api.operators import Op0, Op1
from moptipy.api.process import Process
from moptipy.utils.logger import CSV_SEPARATOR

#: the log section for the frequency table
H_LOG_SECTION: Final[str] = "H"


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
    away from each other. A more general version is available as a fitness
    assignment process (:mod:`~moptipy.algorithms.so.fitness`) that can
    be plugged into a general EA (:mod:`~moptipy.algorithms.so.general_ea`)
    in module :mod:`~moptipy.algorithms.so.fitnesses.ffa`.
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
            if h[new_f] <= best_h:  # frequency of new_f no worse than best_f?
                best_f = new_f  # Store its objective value.
                best_x, new_x = new_x, best_x  # Swap best and new.

        # After we are done, we want to print the H table.
        if h[best_f] == 0:  # Fix the H table for the case that only one
            h[best_f] = 1   # single FE was performed.
        log_h(process, range(len(h)), cast(Callable[[int | float], int],
                                           h.__getitem__),
              cast(Callable[[int | float], str],  # add the lower bound back in
                   lambda i, _lb=lb: str(i + _lb)))


def __h_to_str(indices: Iterable[int | float],
               h: Callable[[int | float], int],
               print_index: Callable[[int | float], str]) -> str:
    """
    Convert a frequency table `H` to a string.

    :param indices: the iterable of indices
    :param h: the history table
    :param print_index: a function to print an index
    :returns: a string representation of the `H` table

    >>> hl = [0, 0, 1, 7, 4, 0, 0, 9, 0]
    >>> __h_to_str(range(len(hl)), hl.__getitem__, str)
    '2;1;3;7;4;4;7;9'
    >>> __h_to_str(range(len(hl)), hl.__getitem__, lambda ii: str(ii + 1))
    '3;1;4;7;5;4;8;9'
    >>> hd = {1: 5, 4: 7, 3: 6, 2: 9}
    >>> __h_to_str(sorted(hd.keys()), hd.__getitem__, str)
    '1;5;2;9;3;6;4;7'
    >>> try:
    ...     hd = {1: 0}
    ...     __h_to_str(sorted(hd.keys()), hd.__getitem__, str)
    ... except ValueError as ve:
    ...     print(ve)
    empty H table?
    """
    first: bool = True
    with StringIO() as out:
        for i in indices:
            v = h(i)
            if v > 0:
                if first:
                    first = False
                else:
                    out.write(CSV_SEPARATOR)
                out.write(print_index(i))
                out.write(CSV_SEPARATOR)
                out.write(str(v))
        if first:
            raise ValueError("empty H table?")
        return out.getvalue()


def log_h(process: Process, indices: Iterable[int | float],
          h: Callable[[int | float], int],
          print_index: Callable[[int | float], str]) -> None:
    """
    Convert a frequency table `H` to a string and log it to a process.

    The frequency table is logged as a single line of text into a section
    `H` delimited by the lines `BEGIN_H` and `END_H`. The line consists
    of `2*n` semi-colon separated values. Each such value pair consists of
    an objective value `y` and its observed frequency `H[y]`. The former is
    either an integer or a float and the latter is an integer.

    :param process: the process
    :param indices: the iterable of indices
    :param h: the history table
    :param print_index: a function to print an index
    """
    if process.has_log():
        s = __h_to_str(indices, h, print_index)
        if len(s) > 0:
            process.add_log_section(H_LOG_SECTION, s)
