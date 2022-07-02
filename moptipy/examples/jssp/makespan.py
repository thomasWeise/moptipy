"""An objective function for minimizing the makespan of `Gantt` charts."""
from typing import Final

import numba  # type: ignore

from moptipy.api.objective import Objective
from moptipy.examples.jssp.gantt import Gantt
from moptipy.examples.jssp.instance import Instance
from moptipy.utils.types import type_error


# start book
@numba.njit(nogil=True, cache=True)
def makespan(x: Gantt) -> int:
    """
    Get the makespan corresponding to a given `Gantt` chart.

    The makespan corresponds to the maximum of the end times of the
    last operation on each machine. This is jitted for performance.

    :param x: the Gantt chart.
    :return: the maximum of any end time stored in the chart
    """
    return int(x[:, -1, 2].max())  # maximum of end time of last op
    # end book


# start book
class Makespan(Objective):
    """Compute the makespan of a `Gantt` chart (for minimization)."""

# end book
    def __init__(self, instance: Instance) -> None:  # +book
        """
        Initialize the makespan objective function.

        :param instance: the instance to load the bounds from
        """
        super().__init__()
        if not isinstance(instance, Instance):
            raise type_error(instance, "instance", Instance)
        self.__instance: Final[Instance] = instance
        #: The fast call forwarding to the makespan function. # +book
        self.evaluate = makespan  # type: ignore # +book

    def lower_bound(self) -> int:
        """
        Get the lower bound of the makespan.

        :return: the lower bound
        """
        return self.__instance.makespan_lower_bound

    def is_always_integer(self) -> bool:
        """
        Return `True` because :func:`makespan` always returns `int` values.

        :retval True: always
        """
        return True

    def upper_bound(self) -> int:
        """
        Get the upper bound of the makespan.

        :return: the sum of all job execution times.
        """
        return self.__instance.makespan_upper_bound

    def __str__(self) -> str:
        """
        Get the name of the makespan objective function.

        :return: `makespan`
        :retval "makespan": always
        """
        return "makespan"
