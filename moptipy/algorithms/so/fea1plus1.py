"""
The FFA-based version of the (1+1)-EA: the (1+1)-FEA.

This algorithm is based on :class:`~moptipy.algorithms.so.rls.RLS`, i.e., the
(1+1)-EA, but uses Frequency Fitness Assignment (FFA) as fitness assignment
process. FFA replaces all objective values with their encounter frequencies in
the selection decisions. The more often an objective value is encountered, the
higher gets its encounter frequency. Therefore, local optima are slowly
receiving worse and worse fitness.

Most of the existing metaheuristic algorithms have in common that they
maintain a set `Si` of one or multiple solutions and derive a set `Ni` of one
or multiple new solutions in each iteration `i`. From the joint set
`Pi = Si + Ni` of old and new solutions, they then select the set `Si+1` of
solutions to be propagated to the next iteration, and so on. This selection
decision is undertaken based mainly on the objective values `f(x)` of the
solutions `x in Pi` and solutions with better objective values tend to be
preferred over solutions with worse objective values.

Frequency Fitness Assignment (FFA) completely breaks with this most
fundamental concept of optimization. FFA was first proposed by
Weise *et al.* as a "plug-in" for metaheuristics intended to prevent
premature convergence. It therefore maintains a frequency table `H` for
objective values. Before the metaheuristic chooses the set `Si+1` from `Pi`,
it increments the encounter frequencies of the objective value of each
solution in `Pi`, i.e., performs `H[yj] <- H[yj] + 1` for each `xj in Pi`,
where `yj = f(xj)`. In its selection decisions, the algorithm then uses the
frequency fitness `H[yj]` instead of the objective values `yj`.

Here we integrate FFA into the randomized local search algorithm
:class:`~moptipy.algorithms.so.rls.RLS`, which is also known as the
`(1+1) EA`. In its original form, RLS maintains a single solution and derives
a slightly modified copy from it in every iteration. If the modified copy is
not worse than the original solution, it replaces it. "Not worse" here means
that its objective value needs to be better or equally good, i.e., `<=`, than
the maintained current best solution. The RLS with FFA (here called
`(1+1) FEA`) now replaces the comparison of objective values with a comparison
of the frequencies of the objective values. Of course, following the
definition of FFA, the frequencies are first incremented (both of the current
and the new solution) and then compared.

The algorithm is here implemented in two different ways: If the objective
function is always integer valued and the difference between its upper and
lower bound is not too high, then we count the frequency fitness by using a
numpy array. This means that frequency updates and getting frequency values is
very fast. If the objective function is not always integer or if the
difference between its maximum and minimum is too large, then we will use
a :class:`collections.Counter` to back the frequency table instead. This will
be slower and probably require more memory, but it may be the only way to
accommodate the frequency table. Of course, this will still fail if there are
too many different objective values and the memory consumed is simply too
high.

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
from collections import Counter
from io import StringIO
from typing import Callable, Final, Iterable, cast

import numpy as np
from numpy.random import Generator

from moptipy.api.algorithm import Algorithm1
from moptipy.api.operators import Op0, Op1
from moptipy.api.process import Process
from moptipy.utils.logger import CSV_SEPARATOR
from moptipy.utils.nputils import DEFAULT_INT
from moptipy.utils.strings import num_to_str

#: the log section for the frequency table
H_LOG_SECTION: Final[str] = "H"

#: The difference between upper- and lower bound at which we switch from
#: using arrays as backing store for the frequency table H to maps.
SWITCH_TO_MAP_RANGE: Final[int] = 67_108_864


def _fea_flat(process: Process, op0: Callable, op1: Callable,
              lb: int, ub: int) -> None:
    """
    Apply the (1+1)-FEA to an optimization problem.

    :param process: the black-box process object
    :param op0: the nullary search operator
    :param op1: the unary search operator
    :param lb: the lower bound
    :param ub: the upper bound
    """
    # Create records for old and new point in the search space.
    best_x = process.create()  # record for best-so-far solution
    new_x = process.create()  # record for new solution

# h holds the encounter frequency of each objective value.
    h: Final[np.ndarray] = np.zeros(ub - lb + 1, DEFAULT_INT)
# Obtain the random number generator.
    random: Final[Generator] = process.get_random()

# Put function references in variables to save time.
    evaluate: Final[Callable] = process.evaluate  # the objective
    should_terminate: Final[Callable] = process.should_terminate

# Start at a random point in the search space and evaluate it.
    op0(random, best_x)  # Create 1 solution randomly and
    best_f: int = cast(int, evaluate(best_x)) - lb  # evaluate it.

    while not should_terminate():  # Until we need to quit...
        op1(random, new_x, best_x)  # new_x = neighbor of best_x
        new_f: int = cast(int, evaluate(new_x)) - lb

        h[new_f] += 1  # Increase the frequency of new_f and
        h[best_f] += 1  # of best_f.
        if h[new_f] <= h[best_f]:  # frequency of new_f no worse than best_f?
            best_f = new_f  # Store its objective value.
            best_x, new_x = new_x, best_x  # Swap best and new.

    # After we are done, we want to print the H table.
    if h[best_f] == 0:  # Fix the H table for the case that only one
        h[best_f] = 1   # single FE was performed.
    log_h(process, range(len(h)), cast(Callable[[int | float], int],
                                       h.__getitem__),
          cast(Callable[[int | float], str],  # add the lower bound back in
               lambda i, _lb=lb: str(i + _lb)))


def _fea_map(process: Process, op0: Callable, op1: Callable) -> None:
    """
    Apply the (1+1)-FEA to an optimization problem.

    :param process: the black-box process object
    :param op0: the nullary search operator
    :param op1: the unary search operator
    """
    # Create records for old and new point in the search space.
    best_x = process.create()  # record for best-so-far solution
    new_x = process.create()  # record for new solution

# h holds the encounter frequency of each objective value.
    h: Final[Counter] = Counter()
# Obtain the random number generator.
    random: Final[Generator] = process.get_random()

# Put function references in variables to save time.
    evaluate: Final[Callable] = process.evaluate  # the objective
    should_terminate: Final[Callable] = process.should_terminate

# Start at a random point in the search space and evaluate it.
    op0(random, best_x)  # Create 1 solution randomly and
    best_f: int | float = evaluate(best_x)  # evaluate it.

    while not should_terminate():  # Until we need to quit...
        op1(random, new_x, best_x)  # new_x = neighbor of best_x
        new_f: int | float = evaluate(new_x)

        h[new_f] += 1  # Increase the frequency of new_f and
        h[best_f] += 1  # of best_f.
        if h[new_f] <= h[best_f]:  # frequency of new_f no worse than best_f?
            best_f = new_f  # Store its objective value.
            best_x, new_x = new_x, best_x  # Swap best and new.

# After we are done, we want to print the H table.
    if h[best_f] == 0:  # Fix the H table for the case that only one
        h[best_f] = 1   # single FE was performed.
    log_h(process, h.keys(),
          cast(Callable[[int | float], int], h.__getitem__), num_to_str)


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
        if process.is_always_integer():
            lb: Final[int | float] = process.lower_bound()
            ub: Final[int | float] = process.upper_bound()
            if isinstance(ub, int) and isinstance(lb, int) \
                    and ((ub - lb) <= SWITCH_TO_MAP_RANGE):
                _fea_flat(process, self.op0.op0, self.op1.op1, lb, ub)
                return
        _fea_map(process, self.op0.op0, self.op1.op1)


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
    >>> hd = Counter({1: 5, 4: 7, 3: 6, 2: 9})
    >>> __h_to_str(sorted(hd.keys()), hd.__getitem__, str)
    '1;5;2;9;3;6;4;7'
    >>> try:
    ...     hd = {1: 0}
    ...     __h_to_str(sorted(hd.keys()), hd.__getitem__, str)
    ... except ValueError as ve:
    ...     print(ve)
    empty H table?
    >>> hx = np.zeros(100, int)
    >>> hx[10] = 4
    >>> hx[12] = 234
    >>> hx[89] = 111
    >>> hx[45] = 2314
    >>> __h_to_str(range(100), hx.__getitem__, str)
    '10;4;12;234;45;2314;89;111'
    >>> __h_to_str(range(100), hx.__getitem__, lambda k: str(10 + k))
    '20;4;22;234;55;2314;99;111'
    >>> hx = np.zeros(100, np.int8)
    >>> hx[10] = 4
    >>> hx[12] = 34
    >>> hx[89] = 11
    >>> hx[45] = 14
    >>> __h_to_str(range(100), hx.__getitem__, str)
    '10;4;12;34;45;14;89;11'
    >>> __h_to_str(range(100), hx.__getitem__, lambda k: str(10 + k))
    '20;4;22;34;55;14;99;11'
    >>> hx = np.zeros(100, np.uint64)
    >>> hx[10] = 4232124356792834738
    >>> hx[12] = 3423443534534
    >>> hx[89] = 13589732857375734566
    >>> hx[45] = 14
    >>> __h_to_str(range(100), hx.__getitem__, str)
    '10;4232124356792834738;12;3423443534534;45;14;89;13589732857375734566'
    >>> __h_to_str(range(100), hx.__getitem__, lambda k: str(10 + k))
    '20;4232124356792834738;22;3423443534534;55;14;99;13589732857375734566'
    """
    with StringIO() as out:
        write: Callable[[str], int] = out.write  # fast call
        csep: Final[str] = CSV_SEPARATOR
        sep: str = ""
        for i in indices:
            v = h(i)
            if v > 0:
                write(sep)
                sep = csep
                write(print_index(i))
                write(sep)
                write(str(v))
        res: Final[str] = out.getvalue()
    if len(res) <= 0:
        raise ValueError("empty H table?")
    return res


def log_h(process: Process, indices: Iterable[int | float],
          h: Callable[[int | float], int],
          print_index: Callable[[int | float], str]) -> None:
    """
    Convert a frequency table `H` to a string and log it to a process.

    The frequency table is logged as a single line of text into a section
    `H` delimited by the lines `BEGIN_H` and `END_H`. The line consists
    of `2*n` semicolon separated values. Each such value pair consists of
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
