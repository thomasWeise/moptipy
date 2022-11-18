"""
A set of numerical optimization algorithms from SciPy.

The function :func:`scipy.optimize.minimize` provides a set of very
efficient numerical/continuous optimization methods. Here we wrap a set of
them into our `moptipy` :class:`~moptipy.api.process.Process` API. All
algorithms provided in this module are imported and wrapped from SciPy
(https://scipy.org).

By using the :func:`~moptipy.api.subprocesses.without_should_terminate`
tool, we can enforce the termination criteria set via the
:class:`~moptipy.api.execution.Execution` builder on external algorithms
while piping all their function evaluations through the
:meth:`~moptipy.api.process.Process.evaluate` routine of the optimization
:meth:`~moptipy.api.process.Process`. This way, we can make these external
algorithms usable within `moptipy` in a transparent manner.
"""
from typing import Any, Callable, Final, cast

from numpy import clip, full, inf, ndarray
from scipy.optimize import Bounds  # type: ignore

# noinspection PyProtectedMember
from scipy.optimize._differentialevolution import (  # type: ignore
    differential_evolution,  # type: ignore
)

# noinspection PyProtectedMember
from scipy.optimize._optimize import (  # type: ignore
    _minimize_bfgs,  # type: ignore
    _minimize_cg,  # type: ignore
    _minimize_neldermead,  # type: ignore
    _minimize_powell,  # type: ignore
)

# noinspection PyProtectedMember
from scipy.optimize._slsqp_py import _minimize_slsqp  # type: ignore

# noinspection PyProtectedMember
from scipy.optimize._tnc import _minimize_tnc  # type: ignore

from moptipy.api.algorithm import Algorithm, Algorithm0
from moptipy.api.operators import Op0
from moptipy.api.process import Process
from moptipy.api.subprocesses import without_should_terminate
from moptipy.operators.vectors.op0_uniform import Op0Uniform
from moptipy.utils.bounds import FloatBounds, OptionalFloatBounds
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.nputils import DEFAULT_FLOAT
from moptipy.utils.types import type_error


# noinspection PyProtectedMember
class SciPyAlgorithmWrapper(Algorithm0, OptionalFloatBounds):
    """
    A wrapper for the Sci-Py API.

    An instance of this class may be re-used, but it must only be used for
    problems of the same dimension.
    """

    def __init__(self, name: str, op0: Op0,
                 min_value: float | None = None,
                 max_value: float | None = None) -> None:
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
        self.__bounds_cache: Bounds | None = None
        #: the cache for starting points
        self.__x0_cache: ndarray | None = None

    def _call(self, func: Callable, x0, max_fes: int,
              bounds: Bounds | None) -> None:
        """
        Invoke the SciPi Algorithm.

        This function will be overwritten to call the SciPi Algorithm.

        :param func: the function to minimize
        :param x0: the starting point
        :param max_fes: the maximum FEs
        :param bounds: the bounds
        """

    def __run(self, pp: Process) -> None:
        """
        Execute the algorithm.

        :param pp: the process
        """
        x0 = self.__x0_cache
        if x0 is None:
            self.__x0_cache = x0 = pp.create()  # create the solution record
            x0_dim = len(x0)  # the dimension of the solution record
            no_bounds: bool = False  # True if no bounds needed
            mi: float = -inf  # the minimum
            ma: float = inf  # the maximum
            if self.min_value is None:
                if self.max_value is None:
                    no_bounds = True  # no bounds needed if none provided
            else:
                mi = self.min_value  # remember minimum (otherwise, mi=-inf)
            if self.max_value is not None:
                ma = self.max_value  # remember maximum (otherwise ma=inf)
            # now create bounds record
            self.__bounds_cache = bounds = None if no_bounds else Bounds(
                full(x0_dim, mi, DEFAULT_FLOAT),  # the lower bound vector
                full(x0_dim, ma, DEFAULT_FLOAT))  # the upper bound vector

        else:  # ok, we have cached bounds (or cached None)
            bounds = self.__bounds_cache

        if bounds is None:
            __func = pp.evaluate
        else:
            def __func(x: ndarray, ff=cast(Callable[[ndarray], Any],
                                           pp.evaluate),
                       lb=self.min_value, ub=self.max_value):
                clip(x, lb, ub, x)
                return ff(x)

        self.op0.op0(pp.get_random(), x0)  # create first solution

        mf = pp.get_max_fes()  # get the number of available FEs
        if mf is not None:  # if an FE limit is specified, then ...
            mf -= pp.get_consumed_fes()  # ... subtract the consumed FEs
        else:  # otherwise set a huge, unattainable limit
            mf = 4_611_686_018_427_387_904  # 2 ** 62
        self._call(__func, x0, mf, bounds)  # invoke the algorithm

    def solve(self, process: Process) -> None:
        """
        Apply the algorithm from SciPy to an optimization problem.

        Basically, this wraps a specific configuration of
        :func:`scipy.optimize.minimize` into our process API and
        invokes it.

        :param process: the black-box process object
        """
        # invoke the SciPy algorithm implementation
        without_should_terminate(
            cast(Callable[[Process], Any], self.__run), process)

    def log_parameters_to(self, logger: KeyValueLogSection):
        """
        Log the parameters of the algorithm to a logger.

        :param logger: the logger for the parameters
        """
        Algorithm0.log_parameters_to(self, logger)  # log algorithm/operator
        OptionalFloatBounds.log_parameters_to(self, logger)  # log bounds


def _call_powell(func: Callable, x0, max_fes: int,
                 bounds: Bounds | None) -> None:
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
       https://doi.org/10.1093/comjnl/7.2.155
    """

    def __init__(self, op0: Op0,
                 min_value: float | None = None,
                 max_value: float | None = None) -> None:
        """
        Create Powell's Algorithm from SciPy.

        :param op0: the nullary search operator
        :param min_value: the minimum x-value
        :param max_value: the maximum x-value
        """
        super().__init__("spPowell", op0, min_value, max_value)
        self._call = _call_powell  # type: ignore


def _call_nelder_mead(func: Callable, x0, max_fes: int,
                      bounds: Bounds | None) -> None:
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
       Applications*. 51(1):259-277. January 2012.
       https://doi.org/10.1007/s10589-010-932
    2. J. A. Nelder and R. Mead. A Simplex Method for Function Minimization.
       *The Computer Journal*. 7(4):308-313. January 1965. Oxford University
       Press (OUP). http://dx.doi.org/10.1093/COMJNL/7.4.308
       https://people.duke.edu/~hpgavin/cee201/Nelder+Mead-\
ComputerJournal-1965.pdf
    3. M. H. Wright. Direct Search Methods: Once Scorned, Now Respectable.
       In D.F. Griffiths and G.A. Watson (Eds.) *Proceedings of the 1995
       Dundee Biennial Conference in Numerical Analysis*. Harlow, UK:
       Addison Wesley Longman, pp. 191-208.
    4. Jorge Nocedal and Stephen J. Wright. *Numerical Optimization*. In
       Springer Series in Operations Research and Financial Engineering.
       New York, NY, USA: Springer. 2006. Second Edition.
       ISBN: 978-0-387-30303-1. Chapter 9.5, Page 238.
       https://doi.org/10.1007/978-0-387-40065-5.
    """

    def __init__(self, op0: Op0,
                 min_value: float | None = None,
                 max_value: float | None = None) -> None:
        """
        Create the Nelder-Mead Algorithm from SciPy.

        :param op0: the nullary search operator
        :param min_value: the minimum x-value
        :param max_value: the maximum x-value
        """
        super().__init__("spNelderMead", op0, min_value, max_value)
        self._call = _call_nelder_mead  # type: ignore


def _call_bgfs(func: Callable, x0, max_fes: int, _) -> None:
    _minimize_bfgs(func, x0, gtol=0.0, maxiter=max_fes)


class BGFS(SciPyAlgorithmWrapper):
    """
    The wrapper for the BGFS algorithm in SciPy.

    This is the quasi-Newton method by C. G. Broyden, Roger Fletcher,
    D. Goldfarb, and David F. Shanno (BFGS).

    1. Jorge Nocedal and Stephen J. Wright. *Numerical Optimization*. In
       Springer Series in Operations Research and Financial Engineering.
       New York, NY, USA: Springer. 2006. Second Edition.
       ISBN: 978-0-387-30303-1. Chapter 6, Page 136.
       https://doi.org/10.1007/978-0-387-40065-5.
    2. Roger Fletcher. *Practical Methods of Optimization* (2nd ed.),
       New York: John Wiley & Sons. 1987. ISBN 978-0-471-91547-8.
    3. C. G. Broyden. The convergence of a class of double-rank minimization
       algorithms. *Journal of the Institute of Mathematics and Its
       Applications*. 6(1):76-90. March 1970.
       http://dx.doi.org/10.1093/imamat/6.1.76
    """

    def __init__(self, op0: Op0,
                 min_value: float | None = None,
                 max_value: float | None = None) -> None:
        """
        Create the BGFS Algorithm from SciPy.

        :param op0: the nullary search operator
        :param min_value: the minimum x-value
        :param max_value: the maximum x-value
        """
        super().__init__("spBgfs", op0, min_value, max_value)
        self._call = _call_bgfs  # type: ignore


def _call_cg(func: Callable, x0, max_fes: int, _) -> None:
    _minimize_cg(func, x0, gtol=0.0, maxiter=max_fes)


class CG(SciPyAlgorithmWrapper):
    """
    The wrapper for the Conjugate Gradient algorithm in SciPy.

    1. Jorge Nocedal and Stephen J. Wright. *Numerical Optimization*. In
       Springer Series in Operations Research and Financial Engineering.
       New York, NY, USA: Springer. 2006. Second Edition.
       ISBN: 978-0-387-30303-1. Chapter 5, Page 101.
    """

    def __init__(self, op0: Op0,
                 min_value: float | None = None,
                 max_value: float | None = None) -> None:
        """
        Create the CG Algorithm from SciPy.

        :param op0: the nullary search operator
        :param min_value: the minimum x-value
        :param max_value: the maximum x-value
        """
        super().__init__("spCg", op0, min_value, max_value)
        self._call = _call_cg  # type: ignore


def _call_slsqp(func: Callable, x0, max_fes: int, _) -> None:
    _minimize_slsqp(func, x0, ftol=0.0, maxiter=max_fes)


class SLSQP(SciPyAlgorithmWrapper):
    """
    The Sequential Least Squares Programming (SLSQP) algorithm in SciPy.

    1. Dieter Kraft. Algorithm 733: TOMP-Fortran modules for optimal control
       calculations. *ACM Transactions on Mathematical Software.*
       20(3):262-281. September 1994. https://doi.org/10.1145/192115.192124
    """

    def __init__(self, op0: Op0,
                 min_value: float | None = None,
                 max_value: float | None = None) -> None:
        """
        Create the SLSQP Algorithm from SciPy.

        :param op0: the nullary search operator
        :param min_value: the minimum x-value
        :param max_value: the maximum x-value
        """
        super().__init__("spSlsqp", op0, min_value, max_value)
        self._call = _call_slsqp  # type: ignore


def _call_tnc(func: Callable, x0, max_fes: int,
              bounds: Bounds | None) -> None:
    if bounds is None:
        b = None
    else:
        b = [(bounds.lb[0], bounds.ub[0])] * len(x0)
    _minimize_tnc(func, x0, bounds=b, ftol=0.0, xtol=0.0, gtol=0.0,
                  maxiter=max_fes, maxfun=max_fes)


class TNC(SciPyAlgorithmWrapper):
    """
    The Truncated Newton Method from SciPy.

    1. Stephen G. Nash. Newton-Type Minimization via the Lanczos Method.
       *SIAM Journal on Numerical Analysis*. 21(4):770-783. August 1984.
       https://dx.doi.org/10.1137/0721052.
    2. Jorge Nocedal and Stephen J. Wright. *Numerical Optimization*. In
       Springer Series in Operations Research and Financial Engineering.
       New York, NY, USA: Springer. 2006. Second Edition.
       ISBN: 978-0-387-30303-1. https://doi.org/10.1007/978-0-387-40065-5.
    """

    def __init__(self, op0: Op0,
                 min_value: float | None = None,
                 max_value: float | None = None) -> None:
        """
        Create the TNC Algorithm from SciPy.

        :param op0: the nullary search operator
        :param min_value: the minimum x-value
        :param max_value: the maximum x-value
        """
        super().__init__("spTnc", op0, min_value, max_value)
        self._call = _call_tnc  # type: ignore


class DE(Algorithm, FloatBounds):
    """
    The Differential Evolution Algorithm as implemented by SciPy.

    At this point, we do not expose the many parameters of the function
    :func:`scipy.optimize.differential_evolution`.
    We only use the default settings. This may change in future releases.

    1. Rainer Storn and Kenneth Price. Differential Evolution - A Simple and
       Efficient Heuristic for global Optimization over Continuous Spaces.
       *Journal of Global Optimization* 11(4):341-359. December 1997.
       https://doi.org/10.1023/A:1008202821328.
       https://www.researchgate.net/publication/227242104
    """

    def __init__(self, dim: int,
                 min_value: float = -1e10,
                 max_value: float = 1e10) -> None:
        """
        Create the Differential Evolution Algorithm from SciPy.

        :param dim: the dimension in which the algorithm works
        :param min_value: the minimum x-value
        :param max_value: the maximum x-value
        """
        FloatBounds.__init__(self, min_value, max_value)
        if not isinstance(dim, int):
            raise type_error(dim, "dim", int)
        if (dim < 1) or (dim > 100_000):
            raise ValueError(
                f"dim must be in 1...100_000 but is {dim}.")
        #: the bounds
        self.__bounds: Final[list[tuple[float, float]]] = \
            [(self.min_value, self.max_value)] * dim

    def __run(self, pp: Process):
        """
        Execute the algorithm.

        :param pp: the process
        """

        def __func(x: ndarray, ff=cast(Callable[[ndarray], Any],
                                       pp.evaluate),
                   lb=self.min_value, ub=self.max_value):
            clip(x, lb, ub, x)
            return ff(x)

        mf = pp.get_max_fes()  # get the number of available FEs
        if mf is not None:  # if an FE limit is specified, then ...
            mf -= pp.get_consumed_fes()  # ... subtract the consumed FEs
        else:  # otherwise set a huge, unattainable limit
            mf = 4_611_686_018_427_387_904  # 2 ** 62

        differential_evolution(
            __func, bounds=self.__bounds,
            maxiter=int(mf / len(self.__bounds)) + 1,
            tol=0.0, seed=pp.get_random(), atol=0.0)

    def solve(self, process: Process) -> None:
        """
        Apply the algorithm from SciPy to an optimization problem.

        Basically, this wraps a specific configuration of
        :func:`scipy.optimize.minimize` into our process API and
        invokes it.

        :param process: the black-box process object
        """
        # invoke the SciPy algorithm implementation
        without_should_terminate(
            cast(Callable[[Process], Any], self.__run), process)

    def log_parameters_to(self, logger: KeyValueLogSection):
        """
        Log the parameters of the algorithm to a logger.

        :param logger: the logger for the parameters
        """
        Algorithm.log_parameters_to(self, logger)  # log algorithm/operator
        FloatBounds.log_parameters_to(self, logger)  # log bounds

    def __str__(self):
        """
        Get the name of this algorithm.

        :returns: the name of this differential evolution algorithm
        """
        return "spDE"
