"""
Fitness Proportionate Selection with Stochastic Uniform Sampling.

In Fitness Proportionate Selection, the chance of a solution for being
selected is proportional to its fitness. This selection scheme is designed for
*maximization*, whereas all optimization in `moptipy` is done as
*minimization*. Therefore, some adjustments are necessary. We will discuss
them later. Let us first introduce the idea of fitness proportionate selection
for maximization.

This idea goes back to the original Genetic Algorithm by Holland. Let us say
that there are `N` elements and the element at index `i` has fitness `v(i)`.
The probability to select this element is then
`P(i) = v(i) / [sum_j=0^N v(j)]`, i.e., the fitness of the element divided by
the overall fitness sum. Let's say we have a population with the fitnesses
`1, 2, 3, 4`. The probability of each element of being selected is then the
fitness divided by the overall fitness sum (here `1+2+3+4=10`), i.e., we get
the probabilities `1/10=0.1, 0.2, 0.3, 0.4`. These nicely add up to `1`.

This can be implemented as follows: First, we copy the fitness values of the
individuals to an array `a` and get, in the above example, `a = [1, 2, 3, 4]`.
Then, we turn this array into a cumulative sum, i.e., add to each element the
sum of all previous elements. We get `a = [1, 3, 6, 10]`. `10`, the last
value, is the overall sum `S` of all fitnesses. Whenever we want to select an
element, we draw a random number `r` uniformly distributed in `[0, S)`. We
perform a binary search to get the right-most insertion index `i` of `r` in
`a`, i.e., the index `i` with `a[i - 1] <= r < a[i]`. Let's say you draw
`0.5`, then `i=0`, for `r=1` you get `i=1`, and for `r=9`, you get `i=3`. `i`
is then the element that has been selected.

In the classical Roulette Wheel Selection as used in Holland's original
Genetic Algorithm, we perform this sampling procedure (draw a random number
`r`, find the index `i` where it belongs, and return the corresponding
element) for each of the `n` offspring we want to sample. An alternative to
this is to perform Stochastic Universal Sampling (SUS) by Baker. Here, the
idea is that we only generate a single random number `r(0)` from within
`[0, S)`. The next number `r(1)` will not be random, but
`r(1)=(r(0) + S/n) mod S` and `r(i) = (r(i-1) + S/n) mod S` where `mod` be
the modulo division. In other words, after drawing the initial random sample,
we take steps of equal length along the Roulette Wheel, or, alternatively,
we have a Roulette Wheel not with a single choice point that we spin `n`
times but a wheel with `n` choice points that we spin a single time. This
avoids random jitter and also requires us to only draw a single random number.
We implement the fitness proportionate selection with stochastic uniform
sampling here.

But this is for *maximization*  of fitness, while we conduct *minimization*.

At first glance, we can turn a maximization problem into a minimization
problem by simply subtracting each fitness value from the maximum fitness
value `max_{all i} v(i)`. However, this has *two* big downsides.

Let us take the same example as before, the fitnesses `1, 2, 3, 4`. Under
maximization, the fitness `4` was best but now it is worst. It is also the
maximum fitness. So let us emulate fitness proportionate selection for
minimization by simply subtracting each fitness from the maximum (here: `4`).
We would get the adjusted fitnesses `3, 2, 1, 0`, the fitness sum `6`, and
probabilities `1/2, 1/3, 1/6, 0`, which still add up to `1` nicely. However,
now we have one element with 0% chance of being selected, namely the one with
the worst original fitness `4`. This is strange, because under maximization,
the worst element was `1` and it still had non-zero probability of being
selected. Furthermore, a second problem occurs if all elements are the same,
say, `1, 1, 1, 1`. We would then adjust them to all zeros, have a zero sum,
and getting `0/0` as selection probabilities. So that is not permissible.

The second problem can easily be solved by simply defining that, if all `N`
elements are identical, they should all get the same probability `1/N` of
being selected.

The first problem becomes clearer when we realize that under "normal" fitness
proportionate selection for maximization, the reference point "`0`" is
actually arbitrarily chosen and hard coded. If we have `1, 2, 3, 4, 10` as
fitnesses and transform them to the probabilities `0.05, 0.1, 0.15, 0.2, 0.5`,
we do so implicitly based on their "distance" to `0`. If we would add
some offset to them, say, `1`, i.e., calculate wit `2, 3, 4, 5, 11`, we would
get the fitness sum `25` and compute probabilities `0.08`, `0.12`, `0.16`,
`0.2`, and `0.44` instead. In other words, if we choose different reference
points, e.g., `-1` instead of `0`, we get different probabilities. And while
`0` seems a natural choice as reference point, it is actually just arbitrary.
The only actual condition of a reference point for maximization is that it
must be less or equal than/to the smallest occurring fitness.

If we do minimization instead of maximization, we do not have a "natural"
reference point. The only condition for the reference point is that it must be
larger or equal than/to the largest occurring fitness. Choosing the maximum
fitness value is just an arbitrary choice and it results in the solution of
this fitness getting `0` chance to reproduce.

If we can choose an arbitrary reference point for minimization, how do we
choose it? Our :class:`~moptipy.algorithms.modules.selections.\
fitness_proportionate_sus.FitnessProportionateSUS` has a parameter
`min_prob`, which corresponds to the minimal selection probability that *any*
element should have (of course, if we have `N` elements in the population, it
must hold that `0 <= min_prob < 1/N`). Based on this probability, we compute
the offset of the fitness values. We do it as follows:

The idea is that, in maximization, we got
`P(i) = v(i) / [sum_j=0^(N-1) v(j)]`. Now if `v(i) = 0`, we would get
`P(i) = 0` as well. But we want `P(i) = min_prob`, so we need to add an
`offset` to each `v(i)`. So this then becomes
`P(i) = min_prob = offset / [sum_j=0^(N-1) (v(j) + offset)]`, which
becomes `min_prob = offset / [N * offset + sum_j=0^N v(j)]`. Let's set
`S = sum_j=0^N v(j)` to make this easier to read and we get
`min_prob = offset / (N * offset + S)`. Solving for `offset` gives us
`offset = S * (min_prob / (1.0 - (min_prob * N)))`. In other words, for any
allowable minimum selection probability `0<=min_prob<1/N`, we can compute an
offset to add to each fitness value that will result in the worst solution
having exactly this selection probability. The probabilities of the other
solutions will be larger, rather proportional to their fitness.

For minimization, first, we compute the maximum (i.e., worst) fitness
`max_fitness` and negate each fitness by subtracting it from `max_fitness`.
For an input array `1, 2, 3, 4` we now get `3, 2, 1, 0`. `S` be the sum of
the negated fitnesses, so in the above example, `S = 3 + 2 + 1 + 0 = 6`. We
can now compute the `offset` to be added to each negated fitness to achieve
the goal probability distribution as follows:
`offset = S * (min_prob / (1.0 - (min_prob * N)))`. If we had chosen
`min_prob = 0`, then `offset = 0` and the probability for the worst element
to be selected remains `0`. If we choose `min_prob = 0.01`, then we would
get `offset = 0.0625`. The selection probability of the worst element with
original fitness `4` and adjusted fitness `0` would be
`(0 + 0.0625) / (6 + (4 * 0.0625)) = 0.0625 / 6.25 = 0.01`.

As a side-effect of this choice of dynamic offsetting, our fitness
proportionate selection scheme becomes invariant under all translations of the
objective function value. The original fitness proportionate selection
schemes, regardless of being of the Roulette Wheel or Stochastic Universal
Sampling variant, do not have this property (see, for instance, de la Maza and
Tidor).

1. John Henry Holland. *Adaptation in Natural and Artificial Systems: An
   Introductory Analysis with Applications to Biology, Control, and Artificial
   Intelligence.* Ann Arbor, MI, USA: University of Michigan Press. 1975.
   ISBN: 0-472-08460-7
2. David Edward Goldberg. *Genetic Algorithms in Search, Optimization, and
   Machine Learning.* Boston, MA, USA: Addison-Wesley Longman Publishing Co.,
   Inc. 1989. ISBN: 0-201-15767-5
3. James E. Baker. Reducing Bias and Inefficiency in the Selection Algorithm.
   In John J. Grefenstette, editor, *Proceedings of the Second International
   Conference on Genetic Algorithms on Genetic Algorithms and their
   Application (ICGA'87),* July 1987, Cambridge, MA, USA, pages 14-21.
   Hillsdale, NJ, USA: Lawrence Erlbaum Associates. ISBN: 0-8058-0158-8
4. Peter J. B. Hancock. An Empirical Comparison of Selection Methods in
   Evolutionary Algorithms. In Terence Claus Fogarty, editor, *Selected Papers
   from the AISB Workshop on Evolutionary Computing (AISB EC'94),* April
   11-13, 1994, Leeds, UK, volume 865 of Lecture Notes in Computer Science,
   pages 80-94, Berlin/Heidelberg, Germany: Springer, ISBN: 978-3-540-58483-4.
   https://dx.doi.org/10.1007/3-540-58483-8_7. Conference organized by the
   Society for the Study of Artificial Intelligence and Simulation of
   Behaviour (AISB).
5. Tobias Blickle and Lothar Thiele. A Comparison of Selection Schemes used in
   Genetic Algorithms. Second edition, December 1995. TIK-Report 11 from the
   Eidgenössische Technische Hochschule (ETH) Zürich, Department of Electrical
   Engineering, Computer Engineering and Networks Laboratory (TIK), Zürich,
   Switzerland. ftp://ftp.tik.ee.ethz.ch/pub/publications/TIK-Report11.ps
6. Uday Kumar Chakraborty and Kalyanmoy Deb and Mandira Chakraborty. Analysis
   of Selection Algorithms: A Markov Chain Approach. *Evolutionary
   Computation,* 4(2):133-167. Summer 1996. Cambridge, MA, USA: MIT Press.
   doi:10.1162/evco.1996.4.2.133.
   https://dl.acm.org/doi/pdf/10.1162/evco.1996.4.2.133
7. Michael de la Maza and Bruce Tidor. An Analysis of Selection Procedures
   with Particular Attention Paid to Proportional and Bolzmann Selection. In
   Stephanie Forrest, editor, *Proceedings of the Fifth International
   Conference on Genetic Algorithms (ICGA'93),* July 17-21, 1993,
   Urbana-Champaign, IL, USA, pages 124-131. San Francisco, CA, USA:
   Morgan Kaufmann Publishers Inc. ISBN: 1-55860-299-2
"""

from math import isfinite
from typing import Any, Callable, Final

import numba  # type: ignore
import numpy as np
from numpy.random import Generator

from moptipy.algorithms.modules.selection import FitnessRecord, Selection
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.nputils import DEFAULT_FLOAT
from moptipy.utils.strings import num_to_str_for_name
from moptipy.utils.types import type_error


@numba.njit(nogil=True, cache=True)
# start book
def _make_cum_sum(a: np.ndarray, offset_mul: float) -> None:
    """
    Compute the roulette wheel based on a given offset multiplier.

    The roulette wheel is basically an array of increasing values which
    corresponds to the cumulative sums of *"maximum - a[i] adjusted with
    the probability offset"*.

    :param a: the array with the fitness values
    :param offset_mul: the offset multiplier

    >>> import numpy as nn
    >>> ar = nn.array([1, 2, 3, 4], float)
    >>> _make_cum_sum(ar, 0)
    >>> list(map(str, ar))
    ['3.0', '5.0', '6.0', '6.0']
    >>> ar = nn.array([1, 2, 3, 4], float)
    >>> min_prob = 0.01
    >>> offset_mult = (min_prob / (1.0 - (min_prob * len(ar))))
    >>> _make_cum_sum(ar, offset_mult)
    >>> list(map(str, ar))
    ['3.0625', '5.125', '6.1875', '6.25']
    >>> (ar[-1] - ar[-2]) / ar[-1]  # compute prob of 4 being selected
    0.01
    >>> ar.fill(12)
    >>> _make_cum_sum(ar, 0.01)
    >>> list(map(str, ar))
    ['1.0', '2.0', '3.0', '4.0']
    """
    max_fitness: float = -np.inf  # initialize maximum to -infinity
    min_fitness: float = np.inf  # initialize minimum to infinity
    for v in a:  # get minimum and maximum fitness
        if v > max_fitness:  # if fitness is bigger than maximum...
            max_fitness = v  # ...then update the maximum
        if v < min_fitness:  # if fitness is smaller than minimum...
            min_fitness = v  # ...then update the minimum

    if min_fitness >= max_fitness:  # all elements are the same
        for i in range(len(a)):  # pylint: disable=C0200
            a[i] = i + 1  # assign equal probabilities to all elements
        return  # finished: a=[1, 2, 3, 4, ...] -> each range = 1

    for i, v in enumerate(a):  # since we do minimization, we now negate
        a[i] = max_fitness - v  # the array by subtracting from maximum

    fitness_sum: Final[float] = a.sum()  # get the new fitness sum

    # compute the offset to accommodate the probability adjustment
    offset: Final[float] = fitness_sum * offset_mul

    cum_sum: float = 0.0  # the cumulative sum accumulator starts at 0
    for i, v in enumerate(a):  # iterate over array and build the sum
        a[i] = cum_sum = cum_sum + offset + v  # store cum sum + offset
# end book


# start book
class FitnessProportionateSUS(Selection):
    """Fitness Proportionate Selection with Stochastic Universal Sampling."""

# end book
    def __init__(self, min_prob: float = 0.0) -> None:
        """
        Create the stochastic universal sampling method.

        :param min_prob: the minimal selection probability of any element
        """
        super().__init__()
        if not isinstance(min_prob, float):
            raise type_error(min_prob, "min_prob", float)
        if not (0.0 <= min_prob < 0.2):
            raise ValueError(
                f"min_prob={min_prob}, but must be 0<=min_prob<0.2")
        #: the minimum selection probability of any element
        self.min_prob: Final[float] = min_prob
        #: the array to store the cumulative sum
        self.__cumsum: np.ndarray = np.empty(0, DEFAULT_FLOAT)

# start book
    def select(self, source: list[FitnessRecord],
               dest: Callable[[FitnessRecord], Any],
               n: int, random: Generator) -> None:
        """
        Perform deterministic best selection without replacement.

        :param source: the list with the records to select from
        :param dest: the destination collector to invoke for each selected
            record
        :param n: the number of records to select
        :param random: the random number generator
        """
        m: Final[int] = len(source)  # number of elements to select from
        # compute the offset multiplier from the minimum probability
        # for this, min_prob must be < 1 / m
        min_prob: Final[float] = self.min_prob
        if min_prob >= (1.0 / m):  # -book
            raise ValueError(f"min_prob={min_prob} >= {1 / m}!")  # -book
        offset_mul: Final[float] = (min_prob / (1.0 - (min_prob * m)))
        # end book
        if (offset_mul < 0.0) or (not isfinite(offset_mul)):
            raise ValueError(
                f"min_prob={min_prob}, len={m} => offset_mul={offset_mul}")

        # start book
        a: np.ndarray = self.__cumsum  # get array for cumulative sum
        # end book
        if len(a) != m:  # re-allocate only if lengths don't match
            self.__cumsum = a = np.empty(m, DEFAULT_FLOAT)
        # start book
        for i, rec in enumerate(source):  # fill the array with fitnesses
            a[i] = rec.fitness  # store the fitnesses in the numpy array

        _make_cum_sum(a, offset_mul)  # construct cumulative sum array
        total_sum: Final[float] = a[-1]  # total sum = last element
        # now perform the stochastic uniform sampling
        current: float = random.uniform(0, total_sum)  # starting point
        step_width: Final[float] = total_sum / n  # step width
        for _ in range(n):  # select the `n` solutions
            dest(source[a.searchsorted(current, "right")])  # select
            current = (current + step_width) % total_sum  # get next
# end book

    def __str__(self):
        """
        Get the name of the stochastic uniform sampling selection algorithm.

        :return: the name of the stochastic uniform sampling selection
            algorithm
        """
        return "fpsus" if self.min_prob <= 0.0 \
            else f"fpsus{num_to_str_for_name(self.min_prob)}"

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of the algorithm to a logger.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        logger.key_value("minprob", self.min_prob, also_hex=True)
