"""
Provides the CMA-ES Family Algorithms from the Library `cmaes`.

The Covariance Matrix Adaptation Evolutionary Strategy, CMA-ES for short, is a
very efficient optimization algorithm for small- and mid-scale and numerical/
continuous optimization problems.

Here, we wrap our `moptipy` API around the beautiful library `cmaes` by
Masashi Shibata and Masahiro Nomura at https://pypi.org/project/cmaes/. They
provide a CMA-ES implementation based on the ask-tell interface. In this
interface, you repeatedly query sample points in the search space from the
model and evaluate them. Then you feed back the points and their corresponding
objective values to the CMA-ES algorithm so that it can update its model. Then
the cycle is repeated.

1. Nikolaus Hansen and Andreas Ostermeier. A Completely Derandomized
   Self-Adaptation in Evolution Strategies. *Evolutionary Computation.*
   9(2):159-195. Summer 2001. https://dx.doi.org/10.1162/106365601750190398
2. Nikolaus Hansen. *The CMA Evolution Strategy: A Tutorial.*
   arXiv:1604.00772, 2016. https://arxiv.org/abs/1604.00772
3. Raymond Ros and Nikolaus Hansen. A Simple Modification in CMA-ES Achieving
   Linear Time and Space Complexity. In Günter Rudolph, Thomas Jansen, Nicola
   Beume, Simon Lucas, and Carlo Poloni, eds., Proceedings of the 10th
   International Conference on Parallel Problem Solving From Nature (PPSN X),
   September 13-17, 2008, Dortmund, Germany, pages 296-305. Volume 5199 of
   Lecture Notes in Computer Science. Berlin/Heidelberg, Germany: Springer.
   http://dx.doi.org/10.1007/978-3-540-87700-4_30
   https://hal.inria.fr/inria-00287367/document
4. Nikolaus Hansen. Benchmarking a BI-Population CMA-ES on the BBOB-2009
   Function Testbed. In Proceedings of the 11th Annual Conference Companion
   on Genetic and Evolutionary Computation Conference: Late Breaking Papers,
   July 8-12, 2009, Montreal, Québec, Canada, pages 2389-2396.
   New York, USA: ACM. http://dx.doi.org/10.1145/1570256.1570333
   https://hal.inria.fr/inria-00382093/document

- https://pypi.org/project/cmaes/
- https://github.com/CyberAgent/cmaes
"""

from typing import Callable, Final

import numpy as np
from cmaes import CMA, SepCMA  # type: ignore
from numpy.random import Generator

from moptipy.api.algorithm import Algorithm
from moptipy.api.process import Process
from moptipy.spaces.vectorspace import VectorSpace
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.types import type_error


def _run_cma(cma: SepCMA | CMA,
             f: Callable[[np.ndarray], int | float],
             should_terminate: Callable[[], bool],
             solutions: list[tuple[np.ndarray, int | float]],
             run_criterion: Callable[[], bool] = lambda: False) -> int:
    """
    Run a CMA implementation from the `cmaes` library.

    This is an internal core routine that translates the ask-tell interface
    of the algorithm implementations in the `cmaes` library into a simple
    loop.

    :param cma: the algorithm instance
    :param f: the objective function
    :param should_terminate: the termination criterion
    :param solutions: the internal list to store the solutions
    :param run_criterion: the stopper for a run
    :returns: the number of consumed FEs if the run was terminated by
      `run_criterion`, `-1` otherwise
    """
    fes: int = 0
    pop_size: Final[int] = cma.population_size

    # now we load a lot of fast call function pointers
    ask: Final[Callable[[], np.ndarray]] = cma.ask
    append: Final[Callable[[
        tuple[np.ndarray, int | float]], None]] = solutions.append
    tell: Final[Callable[
        [list[tuple[np.ndarray, float]]], None]] = cma.tell
    clear: Final[Callable[[], None]] = solutions.clear

    while True:  # the main loop
        clear()  # clear the ask/tell records
        for _ in range(pop_size):
            if should_terminate():  # budget over?
                return -1  # exit
            x: np.ndarray = ask()  # sample a point from CMA-ES
            value: int | float = f(x)  # compute its objective value
            append((x, value))  # store the point
            fes = fes + 1
        tell(solutions)  # feed all results back to the CMA
        if run_criterion():
            return fes


class CMAES(Algorithm):
    """
    A wrapper for the `CMA` algorithm from `cmaes`.

    1. Nikolaus Hansen and Andreas Ostermeier. A Completely Derandomized
       Self-Adaptation in Evolution Strategies. *Evolutionary Computation.*
       9(2):159-195. Summer 2001.
       https://dx.doi.org/10.1162/106365601750190398
    2. Nikolaus Hansen. *The CMA Evolution Strategy: A Tutorial.*
       arXiv:1604.00772, 2016. https://arxiv.org/abs/1604.00772
    """

    def __init__(self, space: VectorSpace) -> None:
        """
        Create the CMAES algorithm.

        :param space: the vector space
        """
        super().__init__()
        if not isinstance(space, VectorSpace):
            raise type_error(space, "space", VectorSpace)
        if space.dimension <= 1:
            raise ValueError("CMA-ES only works on at least two dimensions.")
        #: the vector space defining the dimensions and bounds
        self.space: Final[VectorSpace] = space

    def solve(self, process: Process) -> None:
        """
        Apply the bi-population CMA-ES to an optimization problem.

        :param process: the black-box process object
        """
        f: Final[Callable[[np.ndarray], int | float]] = \
            self.space.clipped(process.evaluate)  # the clipped objective
        should_terminate: Final[Callable[[], bool]] = \
            process.should_terminate  # the termination criterion

        lb: Final[np.ndarray] = self.space.lower_bound  # the upper bound
        ub: Final[np.ndarray] = self.space.upper_bound  # the lower bound
        mean: Final[np.ndarray] = 0.5 * (lb + ub)  # use center as mean value
        sigma: Final[float] = 0.2 * max(ub - lb)  # use a large initial sigma
        bounds: Final[np.ndarray] = \
            np.stack((lb, ub)).transpose()  # construct bounds

        # we create and directly run the CMA-ES algorithm
        _run_cma(CMA(mean=mean, sigma=sigma, bounds=bounds,
                     seed=process.get_random().integers(0, 4294967296)),
                 f, should_terminate, [])

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of the algorithm to a logger.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)  # log algorithm/operator
        self.space.log_bounds(logger)  # log bounds

    def __str__(self):
        """
        Get the name of this optimization algorithm.

        :retval "cmaes_cmaes": always
        """
        return "cmaes_cmaes"


class SepCMAES(CMAES):
    """
    The Separable CMA-ES based on Class `SepCMA` from Library `cmaes`.

    This is a variant of the CMA-ES where the covariance matrix is
    constrained to be diagonal. This means that there are fewer parameters to
    learn, so the learning rate for the covariance matrix can be increased.
    This algorithm is suitable if the problem is of larger scale, i.e., has
    a high dimension, in which case the pure CMA-ES may become rather slow in
    terms of its runtime consumption. Then, the loss of solution quality
    resulting from the underlying assumption that the objective function is
    separable is acceptable versus the gain in speed. By learning only the
    diagonals of the covariance matrix, the implicit assumption is that there
    are no mutual influences between the different decision variables. Of
    course, if the optimization problem is already of that nature, i.e.,
    separable, the algorithm will be faster than the normal CMA-ES at the same
    solution quality.

    1. Raymond Ros and Nikolaus Hansen. A Simple Modification in CMA-ES
       Achieving Linear Time and Space Complexity. In Günter Rudolph,
       Thomas Jansen, Nicola Beume, Simon Lucas, and Carlo Poloni, eds.,
       Proceedings of the 10th International Conference on Parallel
       Problem Solving From Nature (PPSN X), September 13-17, 2008,
       Dortmund, Germany, pages 296-305. Volume 5199 of Lecture Notes in
       Computer Science. Berlin/Heidelberg, Germany: Springer.
       http://dx.doi.org/10.1007/978-3-540-87700-4_30
       https://hal.inria.fr/inria-00287367/document
    """

    def solve(self, process: Process) -> None:
        """
        Apply the separable CMA-ES version to an optimization problem.

        :param process: the optimization problem to solve
        """
        f: Final[Callable[[np.ndarray], int | float]] = \
            self.space.clipped(process.evaluate)  # the clipped objective
        should_terminate: Final[Callable[[], bool]] = \
            process.should_terminate  # the termination criterion

        lb: Final[np.ndarray] = self.space.lower_bound  # the upper bound
        ub: Final[np.ndarray] = self.space.upper_bound  # the lower bound
        mean: Final[np.ndarray] = 0.5 * (lb + ub)  # use center as mean value
        sigma: Final[float] = 0.2 * max(ub - lb)  # use a large initial sigma
        bounds: Final[np.ndarray] = \
            np.stack((lb, ub)).transpose()  # construct bounds

        # we create and directly run the CMA-ES algorithm
        _run_cma(SepCMA(mean=mean, sigma=sigma, bounds=bounds,
                        seed=process.get_random().integers(0, 4294967296)),
                 f, should_terminate, [])

    def __str__(self):
        """
        Get the name of this optimization algorithm.

        :retval "sepCmaes_cmaes": always
        """
        return "sepCmaes_cmaes"


class BiPopCMAES(CMAES):
    """
    The bi-population CMA-ES based on Class `CMA` from Library `cmaes`.

    This algorithm combines two restart strategies for the normal CMA-ES under
    its hood. One where the population size increases exponentially and one
    where varying small population sizes are used.

    We here implement the bi-population CMA-ES algorithm in exactly the same
    way as the authors of the `cmaes` library do on
    https://pypi.org/project/cmaes/.

    1. Nikolaus Hansen. Benchmarking a BI-Population CMA-ES on the BBOB-2009
       Function Testbed. In Proceedings of the 11th Annual Conference
       Companion on Genetic and Evolutionary Computation Conference: Late
       Breaking Papers, July 8-12, 2009, Montreal, Québec, Canada,
       pages 2389-2396. New York, USA: ACM.
       http://dx.doi.org/10.1145/1570256.1570333
       https://hal.inria.fr/inria-00382093/document
    """

    def solve(self, process: Process) -> None:
        """
        Apply the external `cmaes` implementation to an optimization problem.

        :param process: the black-box process object
        """
        f: Final[Callable[[np.ndarray], int | float]] = \
            self.space.clipped(process.evaluate)  # the clipped objective
        should_terminate: Final[Callable[[], bool]] = \
            process.should_terminate  # the termination criterion

        lb: Final[np.ndarray] = self.space.lower_bound  # the upper bound
        ub: Final[np.ndarray] = self.space.upper_bound  # the lower bound
        mean: Final[np.ndarray] = 0.5 * (lb + ub)  # use center as mean value
        sigma: Final[float] = 0.2 * max(ub - lb)  # use a large initial sigma
        bounds: Final[np.ndarray] = \
            np.stack((lb, ub)).transpose()  # construct bounds

        random: Generator = process.get_random()

        # create the initial CMA-ES setup
        cma = CMA(mean=mean, sigma=sigma, bounds=bounds,
                  seed=random.integers(0, 4294967296))

        solutions: list[tuple[np.ndarray, int | float]] = []
        large_pop_restarts: int = 0  # the restarts with big population
        small_pop_fes: int = 0  # the FEs spent in the small population
        large_pop_fes: int = 0  # the FEs spent in the large population
        initial_pop_size: Final[int] = cma.population_size
        is_small_pop: bool = True  # are we in a small-population run?

        # The first run is with the "normal" population size. This is
        # the large population before the first doubling, but its FEs
        # count for the small population.
        while True:  # the main loop
            fes = _run_cma(cma, f, should_terminate, solutions,
                           cma.should_stop)
            if fes < 0:  # this means that should_terminate became True
                return   # so we quit
            if is_small_pop:  # it was a small population so increment
                small_pop_fes += fes  # the small-population-FEs
            else:  # it was a large population, so increment the
                large_pop_fes += fes  # the large-population-FEs

            # We try to spend the same number FEs in small as in the large
            # population.
            is_small_pop = small_pop_fes < large_pop_fes

            if is_small_pop:  # create the small population
                pop_size_multiplier = 2 ** large_pop_restarts
                pop_size = max(1, int(
                    initial_pop_size * pop_size_multiplier ** (
                        random.uniform() ** 2)))
            else:  # else: create the large population
                large_pop_restarts = large_pop_restarts + 1
                pop_size = initial_pop_size * (2 ** large_pop_restarts)

            # Create the new CMA-ES instance.
            cma = CMA(mean=mean, sigma=sigma, bounds=bounds,
                      population_size=pop_size,
                      seed=random.integers(0, 4294967296))

    def __str__(self):
        """
        Get the name of this optimization algorithm.

        :retval "biPopCmaes_cmaes": always
        """
        return "biPopCmaes_cmaes"
