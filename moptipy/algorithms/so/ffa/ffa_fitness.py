"""
The Frequency Fitness Assignment (FFA) Process.

Frequency Fitness Assignment (FFA) replaces all objective values with their
encounter frequencies in the selection decisions. The more often an
objective value is encountered, the higher gets its encounter frequency.
Therefore, local optima are slowly receiving worse and worse fitness.

Notice that this implementation of FFA has a slight twist to it:
It will incorporate the iteration index
(:attr:`~moptipy.algorithms.so.record.Record.it`) of the solutions
into the fitness.
This index is used to break ties, in which case newer solutions are preferred.

This can make the EA with FFA compatible with the
:class:`moptipy.algorithms.so.ffa.fea1plus1.FEA1plus1` if "best" selection
(:class:`moptipy.algorithms.modules.selections.best.Best`) is used
at mu=lambda=1.
To facilitate this, there is one special case in the FFA fitness assignment:
If the population consists of exactly 1 element at iteration index 0, then
the frequency values are not updated.

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
from typing import Final

from numpy.random import Generator
from pycommons.types import type_error, type_name_of

from moptipy.algorithms.so.ffa.ffa_h import clear_h, create_h, log_h
from moptipy.algorithms.so.fitness import Fitness, FRecord
from moptipy.api.objective import Objective, check_objective
from moptipy.api.process import Process
from moptipy.utils.logger import KeyValueLogSection

#: The lower bound at which we switch to an offset-based backing array for
#: the frequency table H.
SWITCH_TO_OFFSET_LB: Final[int] = 8_388_608


class FFA(Fitness):
    """The frequency fitness assignment (FFA) process."""

    def __init__(self, f: Objective, log_h_tbl: bool = False) -> None:
        """
        Create the frequency fitness assignment mapping.

        :param f: the objective function
        """
        super().__init__()
        if not isinstance(log_h_tbl, bool):
            raise type_error(log_h_tbl, "log_h_tbl", bool)
        check_objective(f)
        h, ofs = create_h(f)
        #: the internal H-table
        self.__h: Final = h
        #: the offset
        self.__ofs: Final[int] = ofs
        #: should we log the H-table?
        self.__log_h_tbl = log_h_tbl
        #: are we in the very first iteration?
        self.__first: bool = True

    def __str__(self):
        """
        Get the name (`"ffa"`) of the FFA fitness assignment process.

        :return: the name of this process: `ffa`
        retval "ffa": always
        """
        return "ffa"

    def log_information_after_run(self, process: Process) -> None:
        """Write the H table."""
        if self.__log_h_tbl:
            log_h(process, self.__h, self.__ofs)

    def assign_fitness(self, p: list[FRecord], random: Generator) -> None:
        """
        Assign the frequency fitness.

        :param p: the list of records
        :param random: ignored

        >>> from moptipy.examples.bitstrings.onemax import OneMax
        >>> fit = FFA(OneMax(200))
        >>> a = FRecord(None, 1)
        >>> b = FRecord(None, 2)
        >>> c = FRecord(None, 2)
        >>> d = FRecord(None, 3)
        >>> from numpy.random import default_rng
        >>> rand = default_rng()
        >>> fit.assign_fitness([a, b, c, d], rand)
        >>> assert a.fitness == 1
        >>> assert b.fitness == 2
        >>> assert c.fitness == 2
        >>> assert d.fitness == 1
        >>> fit.assign_fitness([a, b, c, d], rand)
        >>> assert a.fitness == 2
        >>> assert b.fitness == 4
        >>> assert c.fitness == 4
        >>> assert d.fitness == 2
        """
        h: Final = self.__h
        ofs: Final[int] = self.__ofs

        min_it = max_it = p[0].it  # the minimum and maximum iteration index
        for r in p:
            it: int = r.it  # get iteration index from record
            if it < min_it:  # update minimum iteration index
                min_it = it
            elif it > max_it:  # update maximum iteration index
                max_it = it
# Compatibility with (1+1) FEA: In first generation with a population size of
# 1, all elements get the same fitness zero. Otherwise, we would count the
# very first solution in the next fitness assignment step again.
# This creates a small incompatibility between FFA with mu=1 and FFA with m>1.
# This incompatibility is not nice, but it is the only way to ensure that the
# (1+1) FEA and the GeneralEA with EA and mu=1, lambda=1 are identical.
        first = self.__first
        self.__first = False
        if first and (max_it <= 0) and (len(p) <= 1):
            for r in p:  # same fitness: 0
                r.fitness = 0
            return
        for r in p:
            h[r.f + ofs] += 1  # type: ignore
        it_range: Final[int] = max_it - min_it + 1  # range of it index
        for r in p:
            r.fitness = ((int(h[r.f + ofs]) * it_range)  # type: ignore
                         + max_it - r.it)

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log all parameters of this component as key-value pairs.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        logger.key_value("ofs", self.__ofs)
        logger.key_value("hType", type_name_of(self.__h))

    def initialize(self) -> None:
        """Initialize the algorithm."""
        super().initialize()
        clear_h(self.__h)
        self.__first = True
