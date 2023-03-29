"""Some internal helper functions."""

from typing import Final

import numba  # type: ignore
import numpy as np


def _check_max_time_millis(max_time_millis: int | float,
                           total_fes: int | float,
                           total_time_millis: int | float) -> None:
    """
    Check whether a max-time-millis value is permissible.

    If we set a time limit for a run, then the
    :meth:`~moptipy.api.process.Process.should_terminate` will become `True`
    approximately after the time limit has expired. However, this also
    could be later (or maybe even earlier) due to the workings of the
    underlying operating system. And even if
    :meth:`~moptipy.api.process.Process.should_terminate` if `True`, it is not
    clear whether the optimization algorithm can query it right away.
    Instead, it may be blocked in a long-running objective function
    evaluation or some other computation. Hence, it may actually stop
    later. So we cannot simply require that
    `total_time_millis <= max_time_millis`, as this is not practically
    enforceable. Instead, we will heuristically determine a feasible
    maximum limit for how much longer an algorithm might run.

    :param max_time_millis: the max time millis threshold
    :param total_fes: the total FEs performed
    :param total_time_millis: the measured total time millis
    """
    if total_fes == 1:
        return
    div: Final[float] = (total_fes - 2) if total_fes > 1 else 1
    permitted_limit: Final[float] = \
        60_000 + ((1 + (1.1 * (max_time_millis / div))) * total_fes)
    if total_time_millis > permitted_limit:
        raise ValueError(
            f"If max_time_millis is {max_time_millis} and "
            f"total_fes is {total_fes}, then total_time_millis must "
            f"not be more than {permitted_limit}, but is"
            f"{total_time_millis}.")


@numba.njit(nogil=True)
def _get_goal_reach_index(f: np.ndarray, goal_f: int | float):  # noqa
    """
    Compute the offset from the end of `f` when `goal_f` was reached.

    :param f: the raw data array, which must be sorted in
        descending order
    :param goal_f: the goal f value
    :return: the index, or `-1` if `goal_f` was not reached

    >>> ft = np.array([10, 9, 8, 5, 3, 2, 1])
    >>> _get_goal_reach_index(ft, 11)
    0
    >>> ft[_get_goal_reach_index(ft, 11)]
    10
    >>> _get_goal_reach_index(ft, 10)
    0
    >>> _get_goal_reach_index(ft, 9)
    1
    >>> ft[_get_goal_reach_index(ft, 6)]
    5
    >>> _get_goal_reach_index(ft, 1)
    6
    >>> _get_goal_reach_index(ft, 0.9)
    -1
    """
    res: Final = np.searchsorted(f[::-1], goal_f, side="right")
    if res <= 0:
        return -1
    return int(f.size - res)
