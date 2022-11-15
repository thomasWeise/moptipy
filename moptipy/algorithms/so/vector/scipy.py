"""A set of numerical optimization algorithms from SciPy."""
from typing import Final, Callable, Optional, cast, Any

from numpy import full, inf
from scipy.optimize import Bounds  # type: ignore
# noinspection PyProtectedMember
from scipy.optimize._optimize import _minimize_neldermead  # type: ignore
# noinspection PyProtectedMember
from scipy.optimize._optimize import _minimize_powell  # type: ignore

from moptipy.api.algorithm import Algorithm0
from moptipy.api.operators import Op0
from moptipy.api.process import Process
from moptipy.api.subprocesses import without_should_terminate
from moptipy.operators.vectors.op0_uniform import Op0Uniform
from moptipy.utils.bounds import OptionalFloatBounds
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.nputils import DEFAULT_FLOAT


class SciPyAlgorithmWrapper(Algorithm0, OptionalFloatBounds):
    """A wrapper for the Sci-Py API."""

    def __init__(self, name: str, op0: Op0,
                 min_value: Optional[float] = None,
                 max_value: Optional[float] = None) -> None:
        """
        Create the algorithm importer from scipy.

        :param name: the name of the algorithm
        :param op0: the nullary search operator
        :param min_value: the minimum x-value
        :param max_value: the maximum x-value
        """
        Algorithm0.__init__(self, name, op0)
        # load bounds from nullary operator, if possible and not overriden
        if (min_value is None) and (max_value is None) \
                and isinstance(op0, Op0Uniform):
            min_value = op0.min_value
            max_value = op0.max_value
        OptionalFloatBounds.__init__(self, min_value, max_value)
        #: the bounds to be used for the internal function call
        self.__bounds: Optional[Bounds] = None
        #: the bounds cache
        self.__no_bounds_cached: bool = True

    def _call(self, func: Callable, x0, max_fes: int,
              bounds: Optional[Bounds]) -> None:
        """
        Invoke the SciPi Algorithm.

        This function will be overwritten to call the SciPi Algorithm.

        :param func: the function to minimize
        :param x0: the starting point
        :param max_fes: the maximum FEs
        :param bounds: the bounds
        """

    def solve(self, process: Process) -> None:
        """
        Apply the algorithm from SciPy to an optimization problem.

        Basically, this wraps a specific configuration of
        :func:`scipy.optimize.minimize` into our process API and
        invokes it.

        :param process: the black-box process object
        """
        def __run(pp: Process, sf=self):
            x0: Final = pp.create()  # Create the solution record.
            sf.op0.op0(pp.get_random(), x0)  # create first solution
            # create clipped/bounded evaluation method if needed
            mf = pp.get_max_fes()
            if mf is not None:
                mf -= pp.get_consumed_fes()
            else:
                mf = 1_000_000_000_000_000

            x0_dim = len(x0)
            bounds = self.__bounds
            if self.__no_bounds_cached or ((bounds is not None) and
                                           (len(bounds.lb) != x0_dim)):
                if self.min_value is None:
                    if self.max_value is None:
                        return None
                    mi = -inf
                else:
                    mi = self.min_value
                if self.max_value is None:
                    ma = inf
                else:
                    ma = self.max_value
                self.__bounds = bounds = Bounds(
                    full(x0_dim, mi, DEFAULT_FLOAT),
                    full(x0_dim, ma, DEFAULT_FLOAT))
                self.__no_bounds_cached = False
            self._call(pp.evaluate, x0, mf, bounds)

        # invoke the Powell algorithm implementation
        without_should_terminate(
            cast(Callable[[Process], Any], __run), process)

    def log_parameters_to(self, logger: KeyValueLogSection):
        """
        Log the parameters of the algorithm to a logger.

        :param logger: the logger for the parameters
        """
        Algorithm0.log_parameters_to(self, logger)
        OptionalFloatBounds.log_parameters_to(self, logger)


def _call_powell(func: Callable, x0, max_fes: int,
                 bounds: Optional[Bounds]) -> None:
    _minimize_powell(func, x0, bounds=bounds, xtol=0.0, ftol=0.0,
                     maxiter=max_fes, maxfev=max_fes)


class Powell(SciPyAlgorithmWrapper):
    """
    Powell's Algorithm.

    The function :func:`scipy.optimize.minimize` with parameter
    "Powell" can perform unconstrained continuous optimization
    (potentially with boundary constraints)

    1. Michael James David Powell. An Efficient Method for Finding the Minimum
       of a Function of Several Variables without Calculating Derivatives. The
       Computer Journal. 7(2):155-162. 1964.
       doi:https://doi.org/10.1093/comjnl/7.2.155
    """

    def __init__(self, op0: Op0,
                 min_value: Optional[float] = None,
                 max_value: Optional[float] = None) -> None:
        """
        Create Powell's Algorithm from SciPy.

        :param op0: the nullary search operator
        :param min_value: the minimum x-value
        :param max_value: the maximum x-value
        """
        super().__init__("powell", op0, min_value, max_value)
        self._call = _call_powell  # type: ignore


def _call_nelder_mead(func: Callable, x0, max_fes: int,
                      bounds: Optional[Bounds]) -> None:
    _minimize_neldermead(func, x0, bounds=bounds, xatol=0.0, fatol=0.0,
                         maxiter=max_fes, maxfev=max_fes)


class NelderMead(SciPyAlgorithmWrapper):
    """
    The Downhill Simplex aka. the Nelder-Mead Algorithm.

    The function :func:`scipy.optimize.minimize` with parameter
    "Nelder-Mead" can perform unconstrained continuous optimization
    (potentially with boundary constraints) by using the Downhill Simplex
    algorithm a.k.a., the Nelder-Mead algorithm. Here we wrap it into our API.

    Scipy provides the following reference:

    1. Fuchang Gao and Lixing Han. Implementing the Nelder-Mead Simplex
       Algorithm with Adaptive Parameters. *Computational Optimization and
       Applications*. 51(1):259–277. January 2012.
       doi:https://doi.org/10.1007/s10589-010-932
    """

    def __init__(self, op0: Op0,
                 min_value: Optional[float] = None,
                 max_value: Optional[float] = None) -> None:
        """
        Create the Nelder-Mead Algorithm from SciPy.

        :param op0: the nullary search operator
        :param min_value: the minimum x-value
        :param max_value: the maximum x-value
        """
        super().__init__("nelderMead", op0, min_value, max_value)
        self._call = _call_nelder_mead  # type: ignore
