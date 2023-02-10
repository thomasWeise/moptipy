"""
Tournament selection without replacement in the tournaments.

For each slot in the destination, a tournament with
:attr:`~moptipy.algorithms.modules.selections.\
tournament_without_repl.TournamentWithoutReplacement.size`
randomly chosen participating solutions is conducted. The solution with
the best fitness wins and is copied to the destination.

The tournaments are without replacement, following the algorithm by Goldberg,
Korg, and Deb. This means that a solution can only take part in another
tournament after all other solutions have joined one. In other words, we are
drawing solutions without replacement. This means that if we have `m`
solutions and want to select `n` from them by conducting tournaments of size
`s`, each solution will take part in at least `floor(s*n / m)` tournaments and
in at most `ceil(s*n / m)` ones.

We implement this drawing of unique random indices as a partial Fisher-Yates
shuffle. The indices used to choose the tournament contestants from are in an
array forming a permutation. Initially, `unused=m` indices. Whenever we draw
one new index from the permutation, we do so from a random position from
`0..unused-1`, swap it to position `unused-1` and decrement `unused` by one.
Once `unused` reaches `0`, we reset it to `m`. This is different from the
originally proposed implementation of tournament selection without repetition
in that it does not first permutate all the indices. It will be better in
cases where only a few small tournaments are conducted, e.g., during mating
selection. It will be slower when many large tournaments are conducted, e.g.,
during survival selection.

Tournament selection with replacement is implemented in
:mod:`moptipy.algorithms.modules.selections.tournament_with_repl`.

1. David Edward Goldberg and Bradley Korb and Kalyanmoy Deb. Messy Genetic
   Algorithms: Motivation, Analysis, and First Results. *Complex Systems*
   3(5):493-530. 1989.
   https://wpmedia.wolfram.com/uploads/sites/13/2018/02/03-5-5.pdf
2. Kumara Sastry and David Edward Goldberg. Modeling Tournament Selection with
   Replacement using Apparent Added Noise. In Lee Spector, Erik D. Goodman,
   Annie Wu, William Benjamin Langdon, and Hans-Michael Voigt, eds.,
   *Proceedings of the 3rd Annual Conference on Genetic and Evolutionary
   Computation (GECCO'01)*, July 7-11, 2001, San Francisco, CA, USA, page 781.
   San Francisco, CA, United States: Morgan Kaufmann Publishers Inc.
   ISBN: 978-1-55860-774-3. https://dl.acm.org/doi/pdf/10.5555/2955239.2955378
3. Peter J. B. Hancock. An Empirical Comparison of Selection Methods in
   Evolutionary Algorithms. In Terence Claus Fogarty, editor, *Selected Papers
   from the AISB Workshop on Evolutionary Computing (AISB EC'94),* April
   11-13, 1994, Leeds, UK, volume 865 of Lecture Notes in Computer Science,
   pages 80-94, Berlin/Heidelberg, Germany: Springer, ISBN: 978-3-540-58483-4.
   https://dx.doi.org/10.1007/3-540-58483-8_7. Conference organized by the
   Society for the Study of Artificial Intelligence and Simulation of
   Behaviour (AISB).
4. Uday Kumar Chakraborty and Kalyanmoy Deb and Mandira Chakraborty. Analysis
   of Selection Algorithms: A Markov Chain Approach. *Evolutionary
   Computation,* 4(2):133-167. Summer 1996. Cambridge, MA, USA: MIT Press.
   doi:10.1162/evco.1996.4.2.133.
   https://dl.acm.org/doi/pdf/10.1162/evco.1996.4.2.133
5. Tobias Blickle and Lothar Thiele. A Comparison of Selection Schemes used in
   Genetic Algorithms. Second edition, December 1995. TIK-Report 11 from the
   Eidgenössische Technische Hochschule (ETH) Zürich, Department of Electrical
   Engineering, Computer Engineering and Networks Laboratory (TIK), Zürich,
   Switzerland. ftp://ftp.tik.ee.ethz.ch/pub/publications/TIK-Report11.ps
6. Sir Ronald Aylmer Fisher and Frank Yates. *Statistical Tables for
   Biological, Agricultural and Medical Research.* Sixth Edition, March 1963.
   London, UK: Oliver & Boyd. ISBN: 0-02-844720-4. https://digital.library.\
adelaide.edu.au/dspace/bitstream/2440/10701/1/stat_tab.pdf
"""

from math import inf
from typing import Any, Callable, Final

from numpy import empty, ndarray
from numpy.random import Generator

from moptipy.algorithms.modules.selection import FitnessRecord, Selection
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.nputils import DEFAULT_INT, fill_in_canonical_permutation
from moptipy.utils.types import check_int_range


# start book
class TournamentWithoutReplacement(Selection):
    """Tournament selection without replacement in the tournament."""

# end book
    def __init__(self, size: int = 2) -> None:
        """
        Create the tournament selection method.

        :param size: the size of the tournaments
        """
        super().__init__()
        #: the tournament size
        self.size: Final[int] = check_int_range(size, "tournament size", 1)
        #: the cache for the array
        self.__perm: ndarray = empty(0, DEFAULT_INT)

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
        size: Final[int] = self.size  # the tournament size
        m: Final[int] = len(source)  # number of elements to select from
        perm: ndarray = self.__perm  # get the base permutation
        if len(perm) != m:  # -book
            self.__perm = perm = empty(m, DEFAULT_INT)  # -book
            fill_in_canonical_permutation(perm)  # -book
        r0i = random.integers  # fast call to random int from 0..(i-1)
        u: int = m  # the number u of available (unused) indices = m
        for _ in range(n):  # conduct n tournaments
            best: FitnessRecord | None = None  # best competitor
            best_fitness: int | float = inf  # best fitness, initial infinite
            for __ in range(size):  # perform one tournament
                if u == 0:  # if we have used all indices in perm
                    u = m  # then we again at the end
                i = r0i(u)  # get random integer in 0..u-1
                u = u - 1  # decrease number of unused indices
                chosen = perm[i]  # get the index of the chosen element
                perm[u], perm[i] = chosen, perm[u]  # swap to the end
                rec = source[chosen]  # get contestant record
                rec_fitness = rec.fitness  # get its fitness
                if rec_fitness <= best_fitness:  # if better or equal...
                    best = rec  # ... rec becomes the new best record
                    best_fitness = rec_fitness  # and remember fitness
            dest(best)  # at end of the tournament, send best to dest
        # end book

    def __str__(self):
        """
        Get the name of the tournament selection algorithm.

        :return: the name of the tournament selection algorithm
        """
        return f"tour{self.size}"

    def initialize(self) -> None:
        """Initialize this selection algorithm."""
        super().initialize()
        fill_in_canonical_permutation(self.__perm)

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of the algorithm to a logger.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        logger.key_value("size", self.size)
