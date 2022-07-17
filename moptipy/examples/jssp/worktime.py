"""Minimize the sum of the work times of all machines in `Gantt` charts."""
from typing import Final

import numba  # type: ignore

from moptipy.api.objective import Objective
from moptipy.examples.jssp.gantt import Gantt
from moptipy.examples.jssp.instance import Instance
from moptipy.utils.types import type_error


@numba.njit(nogil=True, cache=True)
def worktime(x: Gantt) -> int:
    """
    Get the work time of all machines in a given `Gantt` chart.

    We assume that a machine is turned on when its first job/operation begins
    and is turned off when the last job/operation on it ends.
    During all the time inbetween, the machine is on and a worker needs to be
    present.
    The work time therefore corresponds to the sum of the time units when the
    machines can be turned off minus the time units at which they are turned
    on.

    :param x: the Gantt chart.
    :return: the end times minus the start times
    """
    return int(x[:, -1, 2].sum()) - int(x[:, 0, 2].sum())


class Worktime(Objective):
    """Compute the work time of a `Gantt` chart (for minimization)."""

    def __init__(self, instance: Instance) -> None:  # +book
        """
        Initialize the worktime objective function.

        :param instance: the instance to load the bounds from
        """
        super().__init__()
        if not isinstance(instance, Instance):
            raise type_error(instance, "instance", Instance)
        self.__instance: Final[Instance] = instance
        #: The fast call forwarding to the worktime function. # +book
        self.evaluate = worktime  # type: ignore # +book

    def lower_bound(self) -> int:
        """
        Get the lower bound of the worktime.

        :return: the lower bound
        """
        return self.__instance.makespan_lower_bound \
            + self.__instance.machines - 1

    def is_always_integer(self) -> bool:
        """
        Return `True` because :func:`worktime` always returns `int` values.

        :retval True: always
        """
        return True

    def upper_bound(self) -> int:
        """
        Get the upper bound of the worktime.

        :return: the sum of all job execution times.
        """
        return self.__instance.makespan_upper_bound * self.__instance.machines

    def __str__(self) -> str:
        """
        Get the name of the worktime objective function.

        :return: `worktime`
        :retval "worktime": always
        """
        return "worktime"
