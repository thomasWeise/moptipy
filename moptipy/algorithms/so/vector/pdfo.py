"""
Provides the Algorithm `bobyqa` from the Library `pdfo`.

The library "Powell's Derivative-Free Optimization solvers" (`pdfo`) at
https://www.pdfo.net provides an implementation of the "Bound Optimization BY
Quadratic Approximation" algorithm, or BOBYQA for short.
The library is dedicated to the late Professor M. J. D. Powell FRS (1936—2015)
and maintained by Tom M. Ragonneau and Zaikun Zhang.
Here, we wrap it into a class that complies to our `moptipy` API.
This class offers no additional logic but directly defers to the function
in the `pdfo` library.

1. Michael James David Powell. The BOBYQA Algorithm for Bound Constrained
   Optimization without Derivatives. Technical Report DAMTP 2009/NA06.
   Department of Applied Mathematics and Theoretical Physics, Cambridge
   University, Cambridge, UK, 2009.
   https://www.damtp.cam.ac.uk/user/na/NA_papers/NA2009_06.pdf
2. Tom M. Ragonneau and Zaikun Zhang. *PDFO: a cross-platform package for
   Powell's derivative-free optimization solvers,* arXiv preprint.
   Ithaca, NY, USA: Cornell University Library February, 2023.
   arXiv:2302.13246v1 [math.OC] 26 Feb 202.
   https://arxiv.org/pdf/2302.13246v1

- https://github.com/pdfo/pdfo
- https://www.pdfo.net
- https://pypi.org/project/pdfo
"""

import warnings
from typing import Any, Callable, Final, cast  # pylint: disable=W0611

import numpy as np
import pdfo  # type: ignore
from packaging import version

# noinspection PyProtectedMember
from pdfo._bobyqa import bobyqa  # type: ignore  # noqa: PLC2701
from pycommons.types import type_error

from moptipy.api.algorithm import Algorithm0
from moptipy.api.operators import Op0
from moptipy.api.process import Process
from moptipy.api.subprocesses import (
    get_remaining_fes,
    without_should_terminate,
)
from moptipy.spaces.vectorspace import VectorSpace
from moptipy.utils.logger import KeyValueLogSection


def __check_cannot_use_pdfo() -> bool:
    """
    Check whether we cannot use pdf.

    :returns: `True` if we cannot use pdfo.
    """
    if not hasattr(np, "__version__"):
        return True
    if not hasattr(pdfo, "__version__"):
        return True
    npv: Final[version.Version] = version.parse(np.__version__)
    if npv.major >= 2:
        return True
    pdv: Final[version.Version] = version.parse(pdfo.__version__)
    return ((pdv.major <= 1 <= npv.major) and (npv.minor >= 24)
            and (pdv.minor <= 3))


#: pdfo with version 1.3 and below is incompatible with numpy
#: of version 1.24.0 and above. It will crash with an exception.
#: So for this case, we will later just invoke a single random sample and
#: exit. See https://github.com/pdfo/pdfo/issues/55
#: pdfo with version 2.2 is also incompatible with numpy 2.0 and above.
#: See https://github.com/pdfo/pdfo/issues/112
_CANNOT_DO_PDFO: Final[bool] = __check_cannot_use_pdfo()


class BOBYQA(Algorithm0):
    """
    A wrapper for the `bobyqa` algorithm from `pdfo`.

    The Bound Optimization BY Quadratic Approximation (BOBYQA) developed by
    Michael James David Powell and published by the `pdfo` library.

    1. Michael James David Powell. The BOBYQA Algorithm for Bound Constrained
       Optimization without Derivatives. Technical Report DAMTP 2009/NA06.
       Department of Applied Mathematics and Theoretical Physics, Cambridge
       University, Cambridge, UK, 2009.
       https://www.damtp.cam.ac.uk/user/na/NA_papers/NA2009_06.pdf
    """

    def __init__(self, op0: Op0, space: VectorSpace) -> None:
        """
        Create the BOBYQA algorithm.

        :param op0: the nullary search operator
        :param space: the vector space
        """
        super().__init__("bobyqa_pdfo", op0)
        if not isinstance(space, VectorSpace):
            raise type_error(space, "space", VectorSpace)
        #: the vector space defining the dimensions and bounds
        self.space: Final[VectorSpace] = space

    def __run(self, process: Process) -> None:
        """
        Execute the algorithm.

        :param process: the process
        """
        x0: Final[np.ndarray] = process.create()
        npt: int = (2 * len(x0)) + 1  # the default npt value
        max_fes: int = max(npt + 1, get_remaining_fes(process))
        self.op0.op0(process.get_random(), x0)  # sample start point

        if _CANNOT_DO_PDFO:  # PDFO incompatible to current setup
            process.evaluate(x0)  # do single random sample
            return  # and quit

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bobyqa(self.space.clipped(process.evaluate),
                   x0, bounds=np.stack((self.space.lower_bound,
                                        self.space.upper_bound)).transpose(),
                   options={
                       "rhoend": 1e-320,  # the end rho value
                       "maxfev": max_fes,  # the maximum FEs
                       "npt": npt,  # the number of interpolation points
                       "honour_x0": True,  # enforce our x0
                       "quiet": True})  # do not print any messages

    def solve(self, process: Process) -> None:
        """
        Apply the external `bobyqa` implementation to an optimization problem.

        :param process: the black-box process object
        """
        # invoke the SciPy algorithm implementation
        without_should_terminate(
            cast("Callable[[Process], Any]", self.__run), process)

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of the algorithm to a logger.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)  # log algorithm/operator
        self.space.log_bounds(logger)  # log bounds
