"""
Choose the `n` best elements from the source array.

"Best" selection is the standard way to perform survival selection in a
(mu+lambda) Evolutionary Algorithm (:class:`~moptipy.algorithms.so.ea.EA`).
It basically sorts the input array based on fitness and then copies the first
`n` elements to the output array. The algorithm is not random but
deterministic, i.e., it will always select each of the `n` best records
exactly once.

This selection is also often called "mu+lambda," "mu,lambda," or "top-n"
selection. It is similar to the selection scheme in Eshelman's CHC algorithm.
A randomized version of this algorithm is called "truncation selection" in the
paper of Blickle and Thiele.

1. Peter J. B. Hancock. An Empirical Comparison of Selection Methods in
   Evolutionary Algorithms. In Terence Claus Fogarty, editor, *Selected Papers
   from the AISB Workshop on Evolutionary Computing (AISB EC'94),* April
   11-13, 1994, Leeds, UK, volume 865 of Lecture Notes in Computer Science,
   pages 80-94, Berlin/Heidelberg, Germany: Springer, ISBN: 978-3-540-58483-4.
   https://dx.doi.org/10.1007/3-540-58483-8_7. Conference organized by the
   Society for the Study of Artificial Intelligence and Simulation of
   Behaviour (AISB).
2. Larry J.Eshelman. The CHC Adaptive Search Algorithm: How to Have Safe
   Search When Engaging in Nontraditional Genetic Recombination. In Gregory
   J. E. Rawlins, editor, Foundations of Genetic Algorithms, volume 1, 1991,
   pages 265-283. San Francisco, CA, USA: Morgan Kaufmann.
   https://doi.org/10.1016/B978-0-08-050684-5.50020-3
3. Frank Hoffmeister and Thomas Bäck. Genetic Algorithms and Evolution
   Strategies: Similarities and Differences. In Hans-Paul Schwefel and
   Reinhard Männer, *Proceedings of the International Conference on Parallel
   Problem Solving from Nature (PPSN I),* October 1-3, 1990, Dortmund,
   Germany, volume 496 of Lecture Notes in Computer Science, pages 455-469,
   Berlin/Heidelberg, Germany: Springer. ISBN: 978-3-540-54148-6.
   https://doi.org/10.1007/BFb0029787.
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

from typing import Any, Callable

from numpy.random import Generator

from moptipy.algorithms.modules.selection import FitnessRecord, Selection


# start book
class Best(Selection):
    """The best selection: select each of the best `n` elements once."""

    def select(self, source: list[FitnessRecord],
               dest: Callable[[FitnessRecord], Any],
               n: int, random: Generator) -> None:  # pylint: disable=W0613
        """
        Perform deterministic best selection without replacement.

        :param source: the list with the records to select from
        :param dest: the destination collector to invoke for each selected
            record
        :param n: the number of records to select
        :param random: the random number generator
        """
        source.sort()  # sort by fitness, best solutions come first
        for i in range(n):  # select the n first=best solutions
            dest(source[i])  # by sending them to dest
# end book

    def __str__(self):
        """
        Get the name of the best selection algorithm.

        :return: the name of the best selection algorithm
        """
        return "best"
