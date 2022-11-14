"""The Nelder-Mead algorithm from SciPy, wrapped into our API."""
from typing import Final, Callable, Optional, cast, Any

from scipy.optimize import Bounds  # type: ignore
# noinspection PyProtectedMember
from scipy.optimize._optimize import _minimize_neldermead  # type: ignore

from moptipy.api.algorithm import Algorithm0
from moptipy.api.operators import Op0
from moptipy.api.process import Process
from moptipy.api.subprocesses import without_should_terminate
from moptipy.utils.bounds import to_scipy_bounds, OptionalFloatBounds
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.types import type_error


class NelderMead(Algorithm0, OptionalFloatBounds):
    """The Nelder-Mead Downhill Simplex."""

    def __init__(self, op0: Op0,
                 min_value: Optional[float] = None,
                 max_value: Optional[float] = None,
                 adaptive: bool = False) -> None:
        """
        Create the Nelder-Mead Downhill Simplex..

        :param op0: the nullary search operator
        :param min_value: the minimum x-value
        :param max_value: the maximum x-value
        :param adaptive: is the algorithm adaptive? useful for
            high-dimensional problems
        """
        Algorithm0.__init__(
            self, "nelderMeadA" if adaptive else "nelderMead", op0)
        OptionalFloatBounds.__init__(self, min_value, max_value)

        if not isinstance(adaptive, bool):
            raise type_error(adaptive, "adaptive", bool)
        #: use the adaptive version of Nelder-Mead?
        self.adaptive: Final[bool] = adaptive
        #: the bounds to be used for the internal nelder-mead call
        self.__bounds: Bounds = to_scipy_bounds(self)

    def solve(self, process: Process) -> None:
        """
        Apply the Nelder-Mead algorithm to an optimization problem.

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
            _minimize_neldermead(
                func=pp.evaluate, x0=x0, maxiter=mf, maxfev=mf,
                bounds=self.__bounds, adaptive=sf.adaptive, xatol=0, fatol=0)

        # invoke the Nelder-Mead implementation
        without_should_terminate(
            cast(Callable[[Process], Any], __run), process)

    def log_parameters_to(self, logger: KeyValueLogSection):
        """
        Log the parameters of the algorithm to a logger.

        :param logger: the logger for the parameters
        """
        Algorithm0.log_parameters_to(self, logger)
        OptionalFloatBounds.log_parameters_to(self, logger)  # type: ignore
        logger.key_value("adaptive", self.adaptive)
