"""
Tournament selection with or without replacement in the tournaments.

For each slot in the destination, a tournament with
:attr:`~moptipy.algorithms.modules.selections.tournament.Tournament.s`
randomly chosen participating solutions is conducted. The solution with
the best fitness wins and is copied to the destination. If
:attr:`~moptipy.algorithms.modules.selections.tournament.Tournament\
.replacement` is `True`, then the solutions are drawn with replacement
for each tournament, meaning that one solution may enter a given tournament
several times. If replacements are turned off, each solution may enter each
tournament only at most once. Either way, a solution may be selected multiple
times.

1. Peter J. B. Hancock. An Empirical Comparison of Selection Methods in
   Evolutionary Algorithms. In Terence Claus Fogarty, editor, *Selected Papers
   from the AISB Workshop on Evolutionary Computing (AISB EC'94),* April
   11-13, 1994, Leeds, UK, volume 865 of Lecture Notes in Computer Science,
   pages 80-94, Berlin/Heidelberg, Germany: Springer, ISBN: 978-3-540-58483-4.
   https://dx.doi.org/10.1007/3-540-58483-8_7. Conference organized by the
   Society for the Study of Artificial Intelligence and Simulation of
   Behaviour (AISB).
2. Uday Kumar Chakraborty and Kalyanmoy Deb and Mandira Chakraborty. Analysis
   of Selection Algorithms: A Markov Chain Approach. *Evolutionary
   Computation,* 4(2):133-167. Summer 1996. Cambridge, MA, USA: MIT Press.
   doi:10.1162/evco.1996.4.2.133.
   https://dl.acm.org/doi/pdf/10.1162/evco.1996.4.2.133
3. Tobias Blickle and Lothar Thiele. A Comparison of Selection Schemes used in
   Genetic Algorithms. Second edition, December 1995. TIK-Report 11 from the
   Eidgenössische Technische Hochschule (ETH) Zürich, Department of Electrical
   Engineering, Computer Engineering and Networks Laboratory (TIK), Zürich,
   Switzerland. ftp://ftp.tik.ee.ethz.ch/pub/publications/TIK-Report11.ps
"""

from math import inf
from typing import Any, Callable, Final, Iterable, cast

from numpy.random import Generator

from moptipy.algorithms.modules.selection import FitnessRecord, Selection
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.types import type_error


# start book
class Tournament(Selection):
    """Tournament selection with or without replacement in the tournament."""

# end book
    def __init__(self, s: int = 2, replacement: bool = True) -> None:
        """
        Create the tournament selection method.

        :param s: the size of the tournaments
        :param replacement: will the tournaments be with replacement?
        """
        super().__init__()
        if not isinstance(s, int):
            raise type_error(s, "s", int)
        if s < 1:
            raise ValueError(f"Tournament size must be > 1, but is {s}.")
        if not isinstance(replacement, bool):
            raise type_error(replacement, "replacement", bool)
        #: the tournament size
        self.s: Final[int] = s
        #: should we perform replacements in the tournaments?
        self.replacement: Final[bool] = replacement

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
        # set up tournament size=s, source length=m, replacement
        # such initialization is omitted for the sake of brevity
        # end book
        s: Final[int] = self.s
        replacement: Final[bool] = self.replacement
        m: Final[int] = len(source)

        # fast call
        choice: Final[Callable[[int, int, bool], Iterable[int]]] = \
            cast(Callable[[int, int, bool], Iterable[int]], random.choice)
        # start book
        for _ in range(n):  # conduct n tournaments
            best: FitnessRecord | None = None  # best competitor
            best_fitness: int | float = inf  # best fitness
            for i in choice(m, s, replacement):  # perform tournament
                rec = source[i]  # get record from source
                rec_fitness = rec.fitness  # get its fitness
                if rec_fitness <= best_fitness:  # if better or equal...
                    best = rec  # ... rec becomes the new best record
                    best_fitness = rec_fitness  # and remember fitness
            dest(best)  # at end of the tournament, store best in dest
        # end book

    def __str__(self):
        """
        Get the name of the tournament selection algorithm.

        :return: the name of the tournament selection algorithm
        """
        st = f"tour{self.s}"
        if self.replacement:
            return f"{st}r"
        return st

    def log_parameters_to(self, logger: KeyValueLogSection):
        """
        Log the parameters of the algorithm to a logger.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        logger.key_value("size", self.s)
        logger.key_value("withReplacement", self.replacement)
