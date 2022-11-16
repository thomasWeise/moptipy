"""A set of numerical optimization algorithms from SciPy."""
from typing import Callable, Optional, cast, Any

from numpy import full, inf, ndarray, clip
from scipy.optimize import Bounds  # type: ignore
# noinspection PyProtectedMember
from scipy.optimize._optimize import _minimize_bfgs  # type: ignore
# noinspection PyProtectedMember
from scipy.optimize._optimize import _minimize_cg  # type: ignore
# noinspection PyProtectedMember
from scipy.optimize._optimize import _minimize_neldermead  # type: ignore
# noinspection PyProtectedMember
from scipy.optimize._optimize import _minimize_powell  # type: ignore
# noinspection PyProtectedMember
from scipy.optimize._slsqp_py import _minimize_slsqp  # type: ignore
# noinspection PyProtectedMember
from scipy.optimize._tnc import _minimize_tnc  # type: ignore

from moptipy.api.algorithm import Algorithm0
from moptipy.api.operators import Op0
from moptipy.api.process import Process
from moptipy.api.subprocesses import without_should_terminate
from moptipy.operators.vectors.op0_uniform import Op0Uniform
from moptipy.utils.bounds import OptionalFloatBounds
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.nputils import DEFAULT_FLOAT


# noinspection PyProtectedMember


class SciPyAlgorithmWrapper(Algorithm0, OptionalFloatBounds):
    """
    A wrapper for the Sci-Py API.

    An instance of this class may be re-used, but it must only be used for
    problems of the same dimension.
    """

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
        self.__bounds_cache: Optional[Bounds] = None
        #: the cache for starting points
        self.__x0_cache: Optional[ndarray] = None

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
            x0 = sf.__x0_cache
            if x0 is None:
                sf.__x0_cache = x0 = pp.create()  # create the solution record
                x0_dim = len(x0)  # the dimension of the solution record
                no_bounds: bool = False  # True if no bounds needed
                mi: float = -inf  # the minimum
                ma: float = inf  # the maximum
                if sf.min_value is None:
                    if sf.max_value is None:
                        no_bounds = True  # no bounds needed if none provided
                else:
                    mi = sf.min_value  # remember minimum (otherwise, mi=-inf)
                if sf.max_value is not None:
                    ma = sf.max_value  # remember maximum (otherwise ma=inf)
                # now create bounds record
                sf.__bounds_cache = bounds = None if no_bounds else Bounds(
                    full(x0_dim, mi, DEFAULT_FLOAT),  # the lower bound vector
                    full(x0_dim, ma, DEFAULT_FLOAT))  # the upper bound vector

            else:  # ok, we have cached bounds (or cached None)
                bounds = sf.__bounds_cache

            if bounds is None:
                __func = pp.evaluate
            else:
                def __func(x: ndarray, ff=cast(Callable[[ndarray], Any],
                                               pp.evaluate),
                           lb=sf.min_value, ub=sf.max_value):
                    clip(x, lb, ub, x)
                    return ff(x)

            sf.op0.op0(pp.get_random(), x0)  # create first solution

            mf = pp.get_max_fes()  # get the number of available FEs
            if mf is not None:  # if an FE limit is specified, then ...
                mf -= pp.get_consumed_fes()  # ... subtract the consumed FEs
            else:  # otherwise set a huge, unattainable limit
                mf = 4_611_686_018_427_387_904  # 2 ** 62

            sf._call(__func, x0, mf, bounds)  # invoke the algorithm

        # invoke the Powell algorithm implementation
        without_should_terminate(
            cast(Callable[[Process], Any], __run), process)

    def log_parameters_to(self, logger: KeyValueLogSection):
        """
        Log the parameters of the algorithm to a logger.

        :param logger: the logger for the parameters
        """
        Algorithm0.log_parameters_to(self, logger)  # log algorithm/operator
        OptionalFloatBounds.log_parameters_to(self, logger)  # log bounds


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


def _call_bgfs(func: Callable, x0, max_fes: int, _) -> None:
    _minimize_bfgs(func, x0, gtol=0.0, maxiter=max_fes)


class BGFS(SciPyAlgorithmWrapper):
    """
    The wrapper for the BGFS algorithm in SciPy.

    This is the quasi-Newton method of Broyden, Fletcher, Goldfarb, and
    Shanno (BFGS).
    """

    def __init__(self, op0: Op0,
                 min_value: Optional[float] = None,
                 max_value: Optional[float] = None) -> None:
        """
        Create the BGFS Algorithm from SciPy.

        :param op0: the nullary search operator
        :param min_value: the minimum x-value
        :param max_value: the maximum x-value
        """
        super().__init__("bgfs", op0, min_value, max_value)
        self._call = _call_bgfs  # type: ignore


def _call_cg(func: Callable, x0, max_fes: int, _) -> None:
    _minimize_cg(func, x0, gtol=0.0, maxiter=max_fes)


class CG(SciPyAlgorithmWrapper):
    """The wrapper for the Conjugate Gradient algorithm in SciPy."""

    def __init__(self, op0: Op0,
                 min_value: Optional[float] = None,
                 max_value: Optional[float] = None) -> None:
        """
        Create the BGFS Algorithm from SciPy.

        :param op0: the nullary search operator
        :param min_value: the minimum x-value
        :param max_value: the maximum x-value
        """
        super().__init__("cg", op0, min_value, max_value)
        self._call = _call_cg  # type: ignore


def _call_slsqp(func: Callable, x0, max_fes: int, _) -> None:
    _minimize_slsqp(func, x0, ftol=0.0, maxiter=max_fes)


class SLSQP(SciPyAlgorithmWrapper):
    """The Sequential Least Squares Programming (SLSQP) algorithm in SciPy."""

    def __init__(self, op0: Op0,
                 min_value: Optional[float] = None,
                 max_value: Optional[float] = None) -> None:
        """
        Create the SLSQP Algorithm from SciPy.

        :param op0: the nullary search operator
        :param min_value: the minimum x-value
        :param max_value: the maximum x-value
        """
        super().__init__("slsqp", op0, min_value, max_value)
        self._call = _call_slsqp  # type: ignore


def _call_tnc(func: Callable, x0, max_fes: int,
              bounds: Optional[Bounds]) -> None:
    if bounds is None:
        b = None
    else:
        b = [(bounds.lb[0], bounds.ub[0])] * len(x0)
    _minimize_tnc(func, x0, bounds=b, ftol=0.0, xtol=0.0, gtol=0.0,
                  maxiter=max_fes, maxfun=max_fes)


class TNC(SciPyAlgorithmWrapper):
    """The Truncated Newton Method."""

    def __init__(self, op0: Op0,
                 min_value: Optional[float] = None,
                 max_value: Optional[float] = None) -> None:
        """
        Create the TNC Algorithm from SciPy.

        :param op0: the nullary search operator
        :param min_value: the minimum x-value
        :param max_value: the maximum x-value
        """
        super().__init__("tnc", op0, min_value, max_value)
        self._call = _call_tnc  # type: ignore
