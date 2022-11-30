"""
Tournament selection without replacement in the tournaments.

For each slot in the destination, a tournament with
:attr:`~moptipy.algorithms.modules.selections.\
tournament_without_repl.TournamentWithoutReplacement.size`
randomly chosen participating solutions is conducted. The solution with
the best fitness wins and is copied to the destination. The tournaments are
without replacement, following the algorithm by Goldberg, Korg, and Deb. This
means that A solution can only take part in another tournament after all other
solutions have joined one. In other words, we are drawing solutions without
replacement. This means that if we have `m` solutions and want to select `n`
from them by conducting tournaments of size `s`, each solution will take part
in at least `floor(s*n / m)` tournaments and in at most `ceil(s*n / m)` ones.

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
"""

from math import inf
from typing import Any, Callable, Final

from numpy import array, empty, ndarray
from numpy.random import Generator

from moptipy.algorithms.modules.selection import FitnessRecord, Selection
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.nputils import DEFAULT_INT
from moptipy.utils.types import type_error


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
        if not isinstance(size, int):
            raise type_error(size, "size", int)
        if size < 1:
            raise ValueError(f"Tournament size must be > 1, but is {size}.")
        #: the tournament size
        self.size: Final[int] = size
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
            self.__perm = perm = array(  # -book
                range(m), DEFAULT_INT)  # -book
        shuffle: Final[Callable[[ndarray], None]] = random.shuffle
        pi: int = m  # the current permutation index
        for _ in range(n):  # conduct n tournaments
            best: FitnessRecord | None = None  # best competitor
            best_fitness: int | float = inf  # best fitness, initial infinite
            for __ in range(size):  # perform tournament
                if pi >= m:  # if we have exhausted current randomness
                    shuffle(perm)  # shuffle the permutation randomly
                    pi = 0  # set index in permutation to 0
                rec = source[perm[pi]]  # get contestant record
                pi = pi + 1  # step the permutation index
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

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of the algorithm to a logger.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        logger.key_value("size", self.size)
