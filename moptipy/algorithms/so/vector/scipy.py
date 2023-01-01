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

import numpy as np
from numpy import ndarray
from scipy.optimize import Bounds  # type: ignore

# isort: off
# noinspection PyProtectedMember
from scipy.optimize._differentialevolution import (  # type: ignore
    differential_evolution,  # type: ignore
)
# isort: on

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
from moptipy.api.subprocesses import (
    get_remaining_fes,
    without_should_terminate,
)
from moptipy.spaces.vectorspace import VectorSpace
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.types import type_error


class SciPyAlgorithmWrapper(Algorithm0):
    """
    A wrapper for the Sci-Py API.

    An instance of this class may be re-used, but it must only be used for
    problems of the same dimension.
    """

    def __init__(self, name: str, op0: Op0, space: VectorSpace) -> None:
        """
        Create the algorithm importer from scipy.

        :param name: the name of the algorithm
        :param op0: the nullary search operator
        :param space: the vector space
        """
        super().__init__(name, op0)
        if not isinstance(space, VectorSpace):
            raise type_error(space, "space", VectorSpace)
        #: the vector space defining the dimensions and bounds
        self.space: Final[VectorSpace] = space
        #: the bounds to be used for the internal function call
        self.__bounds: Final[Bounds] = Bounds(space.lower_bound,
                                              space.upper_bound)
        #: the cache for starting points
        self.__x0: Final[ndarray] = space.create()

    def _call(self, func: Callable[[np.ndarray], int | float],
              x0: np.ndarray, max_fes: int, bounds: Bounds) -> None:
        """
        Invoke the SciPi Algorithm.

        This function will be overwritten to call the SciPi Algorithm.

        :param func: the function to minimize
        :param x0: the starting point
        :param max_fes: the maximum FEs
        :param bounds: the bounds
        """

    def __run(self, process: Process) -> None:
        """
        Execute the algorithm.

        :param process: the process
        """
        x0: Final[np.ndarray] = self.__x0
        self.op0.op0(process.get_random(), x0)  # create first solution
        self._call(self.space.clipped(process.evaluate),
                   x0, get_remaining_fes(process),
                   self.__bounds)  # invoke the algorithm

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

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of the algorithm to a logger.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)  # log algorithm/operator
        self.space.log_bounds(logger)  # log bounds


# noinspection PyProtectedMember
def _call_powell(func: Callable[[np.ndarray], int | float],
                 x0: np.ndarray, max_fes: int, bounds: Bounds) -> None:
    _minimize_powell(func, x0, bounds=bounds, xtol=0.0, ftol=0.0,
                     maxiter=max_fes, maxfev=max_fes)


class Powell(SciPyAlgorithmWrapper):
    """
    Powell's Algorithm.

    The function :func:`scipy.optimize.minimize` with parameter
    "Powell" for continuous optimization.

    1. Michael James David Powell. An Efficient Method for Finding the Minimum
       of a Function of Several Variables without Calculating Derivatives. The
       Computer Journal. 7(2):155-162. 1964.
       https://doi.org/10.1093/comjnl/7.2.155
    """

    def __init__(self, op0: Op0, space: VectorSpace) -> None:
        """
        Create Powell's algorithm importer from scipy.

        :param op0: the nullary search operator
        :param space: the vector space
        """
        super().__init__("powell_scipy", op0, space)
        self._call = _call_powell  # type: ignore


def _call_nelder_mead(func: Callable[[np.ndarray], int | float],
                      x0: np.ndarray, max_fes: int, bounds: Bounds) -> None:
    _minimize_neldermead(func, x0, bounds=bounds, xatol=0.0, fatol=0.0,
                         maxiter=max_fes, maxfev=max_fes)


class NelderMead(SciPyAlgorithmWrapper):
    """
    The Downhill Simplex aka. the Nelder-Mead Algorithm.

    The function :func:`scipy.optimize.minimize` with parameter
    "Nelder-Mead" for continuous optimization  by using the Downhill Simplex
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

    def __init__(self, op0: Op0, space: VectorSpace) -> None:
        """
        Create the Nelder-Mead Downhill Simplex importer from scipy.

        :param op0: the nullary search operator
        :param space: the vector space
        """
        super().__init__("nelderMead_scipy", op0, space)
        self._call = _call_nelder_mead  # type: ignore


def _call_bgfs(func: Callable[[np.ndarray], int | float],
               x0: np.ndarray, max_fes: int, _) -> None:
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

    def __init__(self, op0: Op0, space: VectorSpace) -> None:
        """
        Create BGFS algorithm importer from scipy.

        :param op0: the nullary search operator
        :param space: the vector space
        """
        super().__init__("bgfs_scipy", op0, space)
        self._call = _call_bgfs  # type: ignore


def _call_cg(func: Callable[[np.ndarray], int | float],
             x0: np.ndarray, max_fes: int, _) -> None:
    _minimize_cg(func, x0, gtol=0.0, maxiter=max_fes)


class CG(SciPyAlgorithmWrapper):
    """
    The wrapper for the Conjugate Gradient algorithm in SciPy.

    1. Jorge Nocedal and Stephen J. Wright. *Numerical Optimization*. In
       Springer Series in Operations Research and Financial Engineering.
       New York, NY, USA: Springer. 2006. Second Edition.
       ISBN: 978-0-387-30303-1. Chapter 5, Page 101.
    """

    def __init__(self, op0: Op0, space: VectorSpace) -> None:
        """
        Create Conjugate Gradient algorithm importer from scipy.

        :param op0: the nullary search operator
        :param space: the vector space
        """
        super().__init__("cg_scipy", op0, space)
        self._call = _call_cg  # type: ignore


def _call_slsqp(func: Callable[[np.ndarray], int | float],
                x0: np.ndarray, max_fes: int, _) -> None:
    _minimize_slsqp(func, x0, ftol=0.0, maxiter=max_fes)


class SLSQP(SciPyAlgorithmWrapper):
    """
    The Sequential Least Squares Programming (SLSQP) algorithm in SciPy.

    1. Dieter Kraft. Algorithm 733: TOMP-Fortran modules for optimal control
       calculations. *ACM Transactions on Mathematical Software.*
       20(3):262-281. September 1994. https://doi.org/10.1145/192115.192124
    """

    def __init__(self, op0: Op0, space: VectorSpace) -> None:
        """
        Create the SLSQP algorithm importer from scipy.

        :param op0: the nullary search operator
        :param space: the vector space
        """
        super().__init__("slsqp_scipy", op0, space)
        self._call = _call_slsqp  # type: ignore


def _call_tnc(func: Callable[[np.ndarray], int | float],
              x0: np.ndarray, max_fes: int, bounds: Bounds) -> None:
    _minimize_tnc(
        func, x0,
        bounds=[(lb, bounds.ub[i]) for i, lb in enumerate(bounds.lb)],
        ftol=0.0, xtol=0.0, gtol=0.0, maxiter=max_fes, maxfun=max_fes)


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

    def __init__(self, op0: Op0, space: VectorSpace) -> None:
        """
        Create the TNC algorithm importer from scipy.

        :param op0: the nullary search operator
        :param space: the vector space
        """
        super().__init__("tnc_scipy", op0, space)
        self._call = _call_tnc  # type: ignore


class DE(Algorithm):
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

    def __init__(self, space: VectorSpace) -> None:
        """
        Create the Differential Evolution Algorithm from SciPy.

        :param space: the vector space
        """
        super().__init__()
        if not isinstance(space, VectorSpace):
            raise type_error(space, "space", VectorSpace)
        #: the vector space defining the dimensions and bounds
        self.space: Final[VectorSpace] = space
        #: the bounds of the search space, derived from :attr:`space`
        self.__bounds: Final[list[tuple[float, float]]] = \
            [(lb, space.upper_bound[i])
             for i, lb in enumerate(space.lower_bound)]

    def __run(self, process: Process) -> None:
        """
        Execute the algorithm.

        :param process: the process
        """
        mf = get_remaining_fes(process)  # get the number of available FEs

        differential_evolution(
            self.space.clipped(process.evaluate),
            bounds=self.__bounds,
            maxiter=int(mf / len(self.__bounds)) + 1,
            tol=0.0, seed=process.get_random(), atol=0.0)

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

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of the algorithm to a logger.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)  # log algorithm/operator
        with logger.scope("space") as sp:
            self.space.log_parameters_to(sp)  # log space

    def __str__(self):
        """
        Get the name of this algorithm.

        :returns: the name of this differential evolution algorithm
        """
        return "de_scipy"
