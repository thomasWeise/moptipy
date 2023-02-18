"""
An implementation of NSGA-II, a famous multi-objective optimization method.

Here we implement the famous NSGA-II algorithm by Deb, Agrawal, Pratap, and
Meyarivan.

We apply the following modifications to the computation of the crowding
distance:
First, we normalize each dimension based on its range in the population. This
may make sense because some objective functions may have huge ranges while
others have very small ranges and would then basically be ignored in the
crowding distance.
Also, we do not assign infinite crowding distances to border elements of
collapsed dimensions. In other words, if all elements have the same value of
one objective function, none of them would receive infinite crowding distance
for this objective. This also makes sense because in this case, sorting would
be rather random.
We also start by generating the 2*:attr:`NSGA2.pop_size` random solutions.
When selecting parents, we apply tournament selection with replacement.

1. Kalyanmoy Deb, Samir Agrawal, Amrit Pratap, and T. Meyarivan. A Fast
   Elitist Non-Dominated Sorting Genetic Algorithm for Multi-Objective
   Optimization: NSGA-II. In Proceedings of the International Conference
   on Parallel Problem Solving from Nature (PPSN VI), September 18-20, 2000,
   Paris, France, pages 849-858. Lecture Notes in Computer Science,
   volume 1917. ISBN 978-3-540-41056-0. Berlin, Heidelberg: Springer.
   https://doi.org/10.1007/3-540-45356-3_83.
   Also: KanGAL Report Number 200001.
   http://repository.ias.ac.in/83498/1/2-a.pdf.

"""
from math import inf
from typing import Any, Callable, Final, cast

import numpy as np
from numpy.random import Generator

from moptipy.api.algorithm import Algorithm2
from moptipy.api.mo_algorithm import MOAlgorithm
from moptipy.api.mo_archive import MORecord
from moptipy.api.mo_process import MOProcess
from moptipy.api.operators import Op0, Op1, Op2
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.strings import num_to_str_for_name
from moptipy.utils.types import check_int_range, type_error


class _NSGA2Record(MORecord):
    """The NSGA-II specific internal record structure."""

    def __init__(self, x: Any, fs: np.ndarray) -> None:
        """
        Create a multi-objective record.

        :param x: the point in the search space
        :param fs: the vector of objective values
        """
        super().__init__(x, fs)
        #: the list of dominated solutions
        self.dominates: Final[list[_NSGA2Record]] = []
        #: the number of solutions this one is dominated by
        self.n_dominated_by: int = 0
        #: the pareto front index
        self.front: int = -1
        #: the crowding distance
        self.crowding_distance: float = 0.0

    def __lt__(self, other):
        """
        Compare according to domination front and crowding distance.

        :param other: the other element to compare with
        :returns: `True` if an only if this solution has either a smaller
            Pareto front index (:attr:`front`) or the same front index
            and a larger crowding distance (:attr:`crowding_distance`)
        """
        fs: Final[int] = self.front
        fo: Final[int] = other.front
        return (fs < fo) or ((fs == fo) and (
            self.crowding_distance > other.crowding_distance))


def _non_dominated_sorting(
        pop: list[_NSGA2Record],
        needed_at_least: int,
        domination: Callable[[np.ndarray, np.ndarray], int]) -> int:
    """
    Perform the fast non-dominated sorting.

    :param pop: the population
    :param needed_at_least: the number of elements we actually need
    :param domination: the domination relationship
    :returns: the end of the last relevant frontier, with
        `needed_at_least <= end`
    """
    lp: Final[int] = len(pop)

    # clear the domination record
    for rec in pop:
        rec.n_dominated_by = 0

    # compute the domination
    for i in range(lp - 1):
        a = pop[i]
        # we only check each pair of solutions once
        for j in range(i + 1, lp):
            b = pop[j]
            d = domination(a.fs, b.fs)
            if d == -1:  # if a dominates b
                a.dominates.append(b)   # remember that it does
                b.n_dominated_by += 1   # increment b's domination count
            elif d == 1:  # if b dominates a
                b.dominates.append(a)   # remember that it does
                a.n_dominated_by += 1   # increment b's domination count
    # now compute the fronts
    done: int = 0
    front_start: int
    front: int = 0
    while True:  # repeat until all records are done
        front_start = done  # remember end of old front
        front += 1          # step front counter
        for i in range(done, lp):  # process all records not yet in a front
            a = pop[i]  # pick record at index i
            if a.n_dominated_by == 0:  # if it belongs to the current front
                a.front = front  # set its front counter
                pop[i], pop[done] = pop[done], a  # swap it to the front end
                done += 1  # increment length of done records
        if done >= needed_at_least:  # are we done with sorting?
            break  # yes -> break loop and return
        # the new front has been built, it ranges from [old_done, done)
        for j in range(front_start, done):
            a = pop[j]
            for b in a.dominates:
                b.n_dominated_by -= 1
            a.dominates.clear()

    # clear the remaining domination records
    for i in range(front_start, len(pop)):
        pop[i].dominates.clear()

    # return the number of elements sorted
    return done


def _crowding_distances(pop: list[_NSGA2Record]) -> None:
    """
    Compute the crowding distances.

    This method works as in the original NSGA-II, but it normalizes each
    dimension based on their ranges in the population. This may make sense
    because some objective functions may have huge ranges while others have
    very small ranges and would then basically be ignored in the crowding
    distance.

    Also, we do not assign infinite crowding distances to border elements of
    collapsed dimensions. In other words, if all elements have the same value
    of one objective function, none of them would receive infinite crowding
    distance for this objective. This also makes sense because in this case,
    sorting would be rather random.

    :param pop: the population
    """
    for rec in pop:  # set all crowding distances to 0
        rec.crowding_distance = 0.0

    dimensions: Final[int] = len(pop[0].fs)
    for dim in range(dimensions):
        pop.sort(key=cast(Callable[[_NSGA2Record], Any],
                          lambda r, d=dim: r.fs[d]))  # sort by dimension
        first = pop[0]  # get the record with the smallest value in the dim
        last = pop[-1]  # get the record with the largest value in the dim
        a = first.fs[dim]  # get smallest value of dimension
        b = last.fs[dim]   # get largest value of dimension
        if a >= b:    # if smallest >= largest, all are the same!
            continue  # we can ignore the dimension if it has collapsed
        first.crowding_distance = inf  # crowding dist of smallest = inf
        last.crowding_distance = inf   # crowding dist of largest = inf
        div = b - a   # get divisor for normalization
        if div <= 0:  # if the range collapses due to numerical imprecision...
            continue  # then we can also skip this dimension
        # ok, we do have non-zero range
        nex = pop[1]  # for each element, we need those to the left and right
        for i in range(2, len(pop)):
            rec = nex     # first: rec = pop[1], second: pop[2], ....
            nex = pop[i]  # first: nex = pop[2], second: pop[3], ....
            rec.crowding_distance += (nex.fs[dim] - first.fs[dim]) / div
            first = rec   # current rec now = rec to the left next


class NSGA2(Algorithm2, MOAlgorithm):
    """The NSGA-II algorithm by Deb, Agrawal, Pratap, and Meyarivan."""

    def __init__(self, op0: Op0, op1: Op1, op2: Op2,
                 pop_size: int, cr: float) -> None:
        """
        Create the NSGA-II algorithm.

        :param op0: the nullary search operator
        :param op1: the unary search operator
        :param op2: the binary search operator
        :param pop_size: the population size
        :param cr: the crossover rate
        """
        super().__init__(
            f"nsga2_{pop_size}_{num_to_str_for_name(cr)}", op0, op1, op2)
        #: the population size
        self.pop_size: Final[int] = check_int_range(pop_size, "pop_size", 3)
        if not isinstance(cr, float):
            raise type_error(cr, "cr", float)
        if not 0 < cr < 1:
            raise ValueError(f"cr={cr}, but we need 0<cr<1.")
        #: the crossover rate
        self.cr: Final[float] = cr

    def solve_mo(self, process: MOProcess) -> None:
        """
        Apply the MO-RLS to an optimization problem.

        :param process: the black-box process object
        """
        # initialize fast calls and local constants
        create_x: Final[Callable[[], Any]] = process.create
        create_f: Final[Callable[[], np.ndarray]] = process.f_create
        domination: Final[Callable[[np.ndarray, np.ndarray], int]] = \
            process.f_dominates
        should_terminate: Final[Callable[[], bool]] = process.should_terminate
        random: Final[Generator] = process.get_random()
        op0: Final[Callable[[Generator, Any], None]] = self.op0.op0
        op1: Final[Callable[[Generator, Any, Any], None]] = self.op1.op1
        op2: Final[Callable[[Generator, Any, Any, Any], None]] = self.op2.op2
        evaluate: Final[Callable[[Any, np.ndarray], Any]] = process.f_evaluate
        ri: Final[Callable[[int], int]] = random.integers
        rd: Final[Callable[[], float]] = random.uniform
        cr: Final[float] = self.cr

        # create first population
        pop_size: Final[int] = self.pop_size
        pop_size_2: Final[int] = pop_size + pop_size
        pop: list[_NSGA2Record] = []

        # create the population
        for _ in range(pop_size_2):
            if should_terminate():
                process.check_in_all(pop)  # check in population
                return
            rec: _NSGA2Record = _NSGA2Record(create_x(), create_f())
            op0(random, rec.x)
            evaluate(rec.x, rec.fs)
            pop.append(rec)

        while True:  # the main loop
            # Perform non-dominated sorting and get at least pop_size
            # elements in different fronts.
            # "start" marks the begin of the last front that was created.
            # "end" marks the total number of elements sorted.
            # It holds that "start < pop_size <= end".
            end = _non_dominated_sorting(pop, pop_size, domination)
            # We only perform the crowding distance computation on the first
            # "end" elements in the population.
            # We take this slice of the population (or the whole population if
            # all elements are non-domination) and compute the crowding
            # distance.
            candidates = pop[:end] if end < pop_size_2 else pop
            _crowding_distances(candidates)
            # Careful: crowding distance computation will shuffle the list
            # candidates, because it sorts based on dimensions. Therefore, we
            # need to re-sort the whole slice.
            candidates.sort()
            if candidates is not pop:  # write back if need be
                pop[:end] = candidates

            # fill offspring population
            for ofs_idx in range(pop_size, pop_size_2):
                # check if we should leave before evaluation
                if should_terminate():
                    process.check_in_all(pop)
                    return  # quit
                ofs: _NSGA2Record = pop[ofs_idx]  # offspring to overwrite
                # binary tournament with replacement for first parent
                p1: _NSGA2Record = pop[ri(pop_size)]
                palt: _NSGA2Record = pop[ri(pop_size)]
                if palt < p1:
                    p1 = palt

                if rd() >= cr:  # mutation: only 1 parent needed
                    op1(random, ofs.x, p1.x)  # do mutation
                else:  # crossover: need a second parent
                    # binary tournament with replacement for second paren
                    # (who must be different from first parent)
                    p2: _NSGA2Record = p1
                    while p2 is p1:
                        p2 = pop[ri(pop_size)]
                    palt = p1
                    while palt is p1:
                        palt = pop[ri(pop_size)]
                    if palt < p2:
                        p2 = palt
                    # got two parents, now do crossover
                    op2(random, ofs.x, p1.x, p2.x)
                evaluate(ofs.x, ofs.fs)  # otherwise: evaluate

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of the algorithm to a logger.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        logger.key_value("pop_size", self.pop_size)
        logger.key_value("cr", self.cr)

    def initialize(self) -> None:
        """Initialize the algorithm."""
        Algorithm2.initialize(self)
