"""
A (mu+lambda) EA with objective-value identity based clearing.

This is an (mu+lambda) Evolutionary Algorithm with clearing based on the
identity of objective values. This algorithm basically works like a normal
(mu+lambda) EA, but it does not allow solutions with identical objective
values in the population (except in the very first, random generation).
If such solutions occur, then only the newest one is selected. This means
that sometimes, there might be less than `mu` records that are selected,
which then have to produce more than `lambda` offsprings to fill the next
population to size `mu+lambda`.

This `mu+lambda`-EA with this crude clearing works as follows:

1. Start with a list `lst` of `mu` random records and `lambda` blank records.
   Set `n_sel=mu`.

2. In each iteration:

    2.1. Use the `n_sel` first records as input to the search operators to
         generate `mu+lambda-n_sel` new points in the search space.
         For each new point to be created, the binary operator is applied
         with probability `0<=br<=1` and the unary operator is used otherwise.

    2.2. Sort the list `lst` according to the objective value of the record.
         Ties are broken by preferring younger solutions over old ones. Soring
         uses the `__lt__` dunder method of class
         :class:`~moptipy.algorithms.so.record.Record`. This moves the best
         solutions to the front of the list. The tie breaking method both
         encourages drift and ensures compatibility with `RLS`.

    2.3. Purge all records with duplicated objective values from the sorted
         list. Set `n_sel` to be the minimum of `mu` and the number of records
         with unique objective values.

Note 1: During the first iteration, where `mu` random solutions are generated,
no checking for identical objective values is performed. This could be added,
but since this is mainly a convergence prevention method and immediate
convergence from a random sample seems unlikely, I left it as is.

Note 2: This clearing method is much cruder than the one proposed by
Pétrowski. There is no similarity measure or anything, we just use the
identity of objective values to define the niche.

This class here adds the above crude variant of clearing to the implementation
of the Evolutionary Algorithm given in :class:`~moptipy.algorithms.so.ea.EA`.
If `mu=1`, `lambda=1`, and `br=0`, then this algorithm is exactly equivalent
to the :class:`~moptipy.algorithms.so.rls.RLS` if the same unary and nullary
operator are used. It is only a bit slower due to the additional overhead of
maintaining a list of records. This compatibility is achieved by the tie
breaking strategy of `step 2.2` above: RLS will prefer the newer solution over
the current one if the new solution is either better or as same as good. Now
the latter case cannot be achieved by just sorting the list without
considering the iteration at which a solution was created, since sorting in
Python is *stable* (equal elements remain in the order in which they are
encountered in the original list) and because our new solutions would be in
the `lambda` last entries of the list. This can easily be fixed by the tie
breaking, which is implemented in the `__lt__` dunder method of class
:class:`~moptipy.algorithms.so.record.Record`.

1. Alan Pétrowski. A Clearing Procedure as a Niching Method for Genetic
   Algorithms. In Keisoku Jido and Seigyo Gakkai, eds., Proceedings of IEEE
   International Conference on Evolutionary Computation (CEC'96), May 20-22,
   1996, Nagoya, Japan, pages 798-803. Los Alamitos, CA, USA: IEEE Computer
   Society Press, ISBN 0-7803-2902-3. doi:
   https://doi.org/10.1109/ICEC.1996.542703.
2. Alan Pétrowski. An Efficient Hierarchical Clustering Technique for
   Speciation. Evry Cedex, France: Institut National des Télécommunications.
   1997.
3. Thomas Bäck, David B. Fogel, and Zbigniew Michalewicz, eds., *Handbook of
   Evolutionary Computation.* 1997. Computational Intelligence Library.
   New York, NY, USA: Oxford University Press, Inc. ISBN: 0-7503-0392-1
4. James C. Spall. *Introduction to Stochastic Search and Optimization.*
   Estimation, Simulation, and Control - Wiley-Interscience Series in Discrete
   Mathematics and Optimization, volume 6. 2003. Chichester, West Sussex, UK:
   Wiley Interscience. ISBN: 0-471-33052-3. http://www.jhuapl.edu/ISSO/.
"""
from math import inf
from typing import Final, Union, Callable, List, cast, Optional

from numpy.random import Generator

from moptipy.algorithms.so.ea import EA, _int_0, _float_0
from moptipy.algorithms.so.record import Record
from moptipy.api.operators import Op0, Op1, Op2
from moptipy.api.process import Process


class EAwithClearing(EA):
    """
    An `mu+lambda` EA with objective-value identity-based clearing.

    This algorithm works similar to :class:`~moptipy.algorithms.so.ea.EA`,
    but it does not allow the population to contain solutions with identical
    objective values. If two such solutions emerge, only one is retained.
    Therefore, sometimes there might be less than `mu` selected individuals
    and, hence, more than `lambda` offsprings in the next generation.
    """

    def solve(self, process: Process) -> None:
        """
        Apply the EA with clearing to an optimization problem.

        :param process: the black-box process object
        """
        mu: Final[int] = self.mu  # mu: number of best solutions kept
        lst_size: Final[int] = mu + self.lambda_  # size = mu + lambda

        random: Final[Generator] = process.get_random()  # random gen
        create: Final[Callable] = process.create  # create x container
        evaluate: Final[Callable] = process.evaluate  # the objective
        op0: Final[Callable] = self.op0.op0  # the nullary operator
        op1: Final[Callable] = self.op1.op1  # the unary operator
        op2: Final[Callable] = self.op2.op2  # the binary operator
        br: Final[float] = self.br  # the rate at which to use op2
        should_terminate: Final[Callable] = process.should_terminate
        r0i: Final[Callable[[int], int]] = cast(  # only if m > 1, we
            Callable[[int], int], random.integers  # need random
            if mu > 1 else _int_0)  # indices
        r01: Final[Callable[[], float]] = cast(  # only if 0<br<1, we
            Callable[[], float],  # need random floats
            random.random if 0 < br < 1 else _float_0)

        # create list of mu random records and lambda empty records
        lst: Final[List] = [None] * lst_size  # pre-allocate list
        f: Union[int, float] = 0  # variable to hold objective values
        n_sel: int = mu  # the number of selected parents
        for i in range(lst_size):  # fill list of size mu+lambda
            x = create()  # by creating point in search space
            if i < n_sel:  # only the first mu records are initialized
                op0(random, x)  # apply nullary operator = randomize
                if should_terminate():  # should we quit?
                    return  # computational budget exhausted -> quit
                f = evaluate(x)  # continue? ok, evaluate new solution
            lst[i] = Record(x, f)  # create and store record

        it: int = 0
        while True:  # keep 0..n_sel-1, overwrite n_sel..mu+lambda-1
            it += 1  # step iteration counter
            for oi in range(n_sel, lst_size):  # for all offspring
                if should_terminate():  # only continue if we still...
                    return  # have sufficient budget ... otherwise quit
                dest: Record = lst[oi]  # pick destination record
                x = dest.x  # the destination "x" value
                dest.it = it  # remember iteration of solution creation

                sx = lst[r0i(n_sel)].x  # pick a random source record
                if (n_sel > 1) and (r01() < br):  # apply binary op?
                    sx2 = sx  # second source "x"
                    while sx2 is sx:  # must be different from sx
                        sx2 = lst[r0i(n_sel)].x  # get second record
                    op2(random, x, sx, sx2)  # apply binary op
                else:
                    op1(random, x, sx)  # apply unary operator
                dest.f = evaluate(x)  # evaluate new point

            lst.sort()  # best records come first, ties broken by age

            forbidden_f = -inf  # the last objective value used
            n_sel = 0  # the number of selected records
            for cur_idx, cur_rec in enumerate(lst):  # enumerate list
                cur_f = cur_rec.f  # get the current objective value
                if cur_f > forbidden_f:  # otherwise, it's a repetition
                    if cur_idx > n_sel:  # if yes, we skipped records
                        lst[cur_idx] = lst[n_sel]  # and need to swap
                        lst[n_sel] = cur_rec  # the selected one in
                    n_sel += 1  # we got a new unique objective value
                    forbidden_f = cur_f  # so let's forbid repeating it
                    if n_sel >= mu:  # did we get mu unique ones?
                        break  # great, we can stop

    def __init__(self, op0: Op0,
                 op1: Optional[Op1] = None,
                 op2: Optional[Op2] = None,
                 mu: int = 1, lambda_: int = 1,
                 br: Optional[float] = None) -> None:
        """
        Create the Evolutionary Algorithm (EA).

        :param op0: the nullary search operator
        :param op1: the unary search operator
        :param op2: the binary search operator
        :param mu: the number of best solutions to survive in each generation
        :param lambda_: the number of offspring in each generation
        :param br: the rate at which the binary operator is applied
        """
        super().__init__(op0, op1, op2, mu, lambda_, br, "eac")
