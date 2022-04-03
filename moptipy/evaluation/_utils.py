"""Some internal helper functions."""

from typing import Union, Final

import numba  # type: ignore
import numpy as np

import moptipy.api.logging as lg

#: the maximum FEs of a black-box process
_FULL_KEY_MAX_FES: Final[str] = f"{lg.SCOPE_PROCESS}.{lg.KEY_MAX_FES}"
#: the maximum runtime in milliseconds of a black-box process
_FULL_KEY_MAX_TIME_MILLIS: Final[str] = \
    f"{lg.SCOPE_PROCESS}.{lg.KEY_MAX_TIME_MILLIS}"
#: the goal objective value of a black-box process
_FULL_KEY_GOAL_F: Final[str] = f"{lg.SCOPE_PROCESS}.{lg.KEY_GOAL_F}"
#: the random seed
_FULL_KEY_RAND_SEED: Final[str] = f"{lg.SCOPE_PROCESS}.{lg.KEY_RAND_SEED}"
#: the FE when the best objective value was reached


def _check_max_time_millis(max_time_millis: Union[int, float],
                           total_fes: int,
                           total_time_millis: int) -> None:
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
def _get_reach_index(f: np.ndarray, goal_f: Union[int, float]):
    """
    Compute the offset from the end of `f` when `goal_f` was reached.

    :param f: the raw data array, which must be sorted in
        descending order
    :param goal_f: the goal f value
    :return: the index, or `0` if `goal_f` was not reached

    >>> ft = np.array([10, 9, 8, 5, 3, 2, 1])
    >>> _get_reach_index(ft, 11)
    7
    >>> ft[ft.size - _get_reach_index(ft, 11)]
    10
    >>> _get_reach_index(ft, 10)
    7
    >>> _get_reach_index(ft, 9)
    6
    >>> ft[ft.size - _get_reach_index(ft, 6)]
    5
    >>> _get_reach_index(ft, 1)
    1
    >>> _get_reach_index(ft, 0.9)
    0
    """
    return np.searchsorted(f[::-1], goal_f, side="right")
