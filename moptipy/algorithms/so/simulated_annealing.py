"""
The simulated annealing algorithm with configurable temperature schedule.

A basic randomized local search (:mod:`~moptipy.algorithms.so.rls`) maintains
one interesting solution and derives one new solution from it using the
unary search operator (:class:`~moptipy.api.operators.Op1`). The new solution
replaces the current solution if it is not worse, i.e., better or equally
good. Simulated Annealing is similar to the :mod:`~moptipy.algorithms.so.rls`,
but it sometimes also accepts solutions that are worse than the current one.
It does so with a probability that becomes smaller the worse the new solution
is and also becomes smaller the smaller the current "temperature" is.

Simulated Annealing applies a so-called temperature schedule (see
:mod:`~moptipy.algorithms.modules.temperature_schedule`), which basically is
function that relates the index of the algorithm iteration (i.e., the index
of the current objective function evaluation) to a temperature. It therefore
is a function accepting an integer value as input and returning a float
temperature. This function is usually monotonously decreasing over time,
meaning that the initial "temperature" is high and then becomes smaller. The
algorithm therefore is more likely to accept worse solutions at its beginning,
whereas it behaves basically like a :mod:`~moptipy.algorithms.so.rls` at the
end of its computational budget (if configured correctly).

Simulated Annealing was independently developed by several researchers [1-4].
The idea is inspired by Metropolis' approximation of how annealing can be
simulated [5].

1. Scott Kirkpatrick, C. Daniel Gelatt, Jr., and Mario P. Vecchi. Optimization
   by Simulated Annealing. *Science Magazine.* 220(4598):671-680.
   May 13, 1983. doi: https://doi.org/10.1126/science.220.4598.671.
   https://www.researchgate.net/publication/6026283
2. Vladimír Černý. Thermodynamical Approach to the Traveling Salesman Problem:
   An Efficient Simulation Algorithm. *Journal of Optimization Theory and
   Applications.* 45(1):41-51. January 1985.
   doi: https://doi.org/10.1007/BF00940812.
   http://mkweb.bcgsc.ca/papers/cerny-travelingsalesman.pdf.
3. Dean Jacobs, Jan Prins, Peter Siegel, and Kenneth Wilson. Monte Carlo
   Techniques in Code Optimization. *ACM SIGMICRO Newsletter.* 13(4):143-148.
   December 1982. Also in Proceedings of the 15th Annual Workshop on
   Microprogramming (MICRO 15), October 5-7, 1982, Palo Alto, CA, USA,
   New York, NY, USA: ACM. doi: http://doi.org/10.1145/1014194.800944.
4. Martin Pincus. Letter to the Editor - A Monte Carlo Method for the
   Approximate Solution of Certain Types of Constrained Optimization Problems.
   *Operations Research.* 18(6):1225-1228. November/December 1970.
   doi: https://doi.org/10.1287/opre.18.6.1225.
5. Nicholas Metropolis, Arianna W. Rosenbluth, Marshall Nicholas Rosenbluth,
   Augusta H. Teller, Edward Teller. Equation of State Calculations by Fast
   Computing Machines. *The Journal of Chemical Physics*. 21(6):1087-1092.
   June 1953. doi: https://doi.org/10.1063/1.1699114.
   http://scienze-como.uninsubria.it/bressanini/montecarlo-history/mrt2.pdf.
"""
from math import exp
from typing import Callable, Final

from numpy.random import Generator

from moptipy.algorithms.modules.temperature_schedule import TemperatureSchedule
from moptipy.api.algorithm import Algorithm1
from moptipy.api.operators import Op0, Op1
from moptipy.api.process import Process
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.types import type_error


# start book
class SimulatedAnnealing(Algorithm1):
    """Simulated Annealing is an RLS sometimes accepting worsening moves."""

    def solve(self, process: Process) -> None:
        """
        Apply the simulated annealing algorithm to an optimization problem.

        :param process: the black-box process object
        """
        # Create records for old and new point in the search space.
        best_x = process.create()  # record for best-so-far solution
        new_x = process.create()  # record for new solution
        # Obtain the random number generator.
        random: Final[Generator] = process.get_random()

        # Put function references in variables to for faster calls.
        temperature: Final[Callable[[int], float]] = self.schedule.temperature
        r01: Final[Callable[[], float]] = random.random  # random from [0, 1)
        evaluate: Final[Callable] = process.evaluate  # the objective
        op1: Final[Callable] = self.op1.op1  # the unary operator
        should_terminate: Final[Callable] = process.should_terminate

        # Start at a random point in the search space and evaluate it.
        self.op0.op0(random, best_x)  # Create 1 solution randomly and
        best_f: int | float = evaluate(best_x)  # evaluate it.
        tau: int = 0  # The iteration index, needs to be 0 at first cmp.

        while not should_terminate():  # Until we need to quit...
            op1(random, new_x, best_x)  # new_x = neighbor of best_x
            new_f: int | float = evaluate(new_x)
            if (new_f <= best_f) or (  # Accept if <= or if SA criterion
                    r01() < exp((best_f - new_f) / temperature(tau))):
                best_f = new_f  # Store its objective value.
                best_x, new_x = new_x, best_x  # Swap best and new.
            tau = tau + 1  # Step the iteration index.
# end book

    def __init__(self, op0: Op0, op1: Op1,
                 schedule: TemperatureSchedule) -> None:
        """
        Create the simulated annealing algorithm.

        :param op0: the nullary search operator
        :param op1: the unary search operator
        :param schedule: the temperature schedule to use
        """
        super().__init__(f"sa_{schedule}", op0, op1)
        if not isinstance(schedule, TemperatureSchedule):
            raise type_error(schedule, "schedule", TemperatureSchedule)
        #: the temperature schedule
        self.schedule: Final[TemperatureSchedule] = schedule

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log all parameters of the simulated annealing algorithm.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        with logger.scope("ts") as ts:
            self.schedule.log_parameters_to(ts)

    def initialize(self) -> None:
        """Initialize the algorithm."""
        super().initialize()
        self.schedule.initialize()
