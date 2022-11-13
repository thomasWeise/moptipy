"""The Nelder-Mead algorithm from SciPy, wrapped into our API."""
from math import isfinite
from typing import Final, Callable, Optional, cast, Any

# noinspection PyProtectedMember
from scipy.optimize._optimize import _minimize_neldermead  # type: ignore
from scipy.optimize import Bounds  # type: ignore

from moptipy.api.algorithm import Algorithm0
from moptipy.api.operators import Op0
from moptipy.api.process import Process
from moptipy.api.subprocesses import without_should_terminate
from moptipy.utils.types import type_error
from moptipy.utils.logger import KeyValueLogSection


class NelderMead(Algorithm0):
    """The Nelder-Mead Downhill Simplex."""

    def __init__(self, op0: Op0,
                 x_min: Optional[float] = None,
                 x_max: Optional[float] = None,
                 adaptive: bool = False) -> None:
        """
        Create the Nelder-Mead Downhill Simplex..

        :param op0: the nullary search operator
        :param x_min: the minimum x-value
        :param x_max: the maximum x-value
        :param adaptive: is the algorithm adaptive? useful for
            high-dimensional problems
        """
        super().__init__("nelderMeadA" if adaptive else "nelderMead", op0)

        if x_min is not None:
            if not isinstance(x_min, float):
                raise type_error(x_min, "x_min", float)
            if not isfinite(x_min):
                raise ValueError(f"x_min must be finite, but is {x_min}.")
        if x_max is not None:
            if not isinstance(x_max, float):
                raise type_error(x_max, "x_max", float)
            if not isfinite(x_max):
                raise ValueError(f"x_max must be finite, but is {x_max}.")
            if (x_min is not None) and (x_min >= x_max):
                raise ValueError(f"x_max > x_min must hold, but got "
                                 f"x_min={x_min} and x_max={x_max}.")
        if not isinstance(adaptive, bool):
            raise type_error(adaptive, "adaptive", bool)
        #: the minimum permitted coordinate value, or None for no lower bound
        self.x_min: Final[Optional[float]] = x_min
        #: the maximum permitted coordinate value, or None for no upper bound
        self.x_max: Final[Optional[float]] = x_max
        #: use the adaptive version of Nelder-Mead?
        self.adaptive: Final[bool] = adaptive

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
                bounds=None if (sf.x_min is None) and (sf.x_max is None) else
                Bounds(sf.x_min, sf.x_max),
                adaptive=sf.adaptive, xatol=0, fatol=0)

        # invoke the Nelder-Mead implementation
        without_should_terminate(
            cast(Callable[[Process], Any], __run), process)

    def log_parameters_to(self, logger: KeyValueLogSection):
        """
        Log the parameters of the algorithm to a logger.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        logger.key_value("xMin", self.x_min, also_hex=True)
        logger.key_value("xMax", self.x_max, also_hex=True)
        logger.key_value("adaptive", self.adaptive)
