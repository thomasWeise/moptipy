"""An implementation of the operation-based encoding for JSSPs."""
from typing import Final

import numba  # type: ignore
import numpy as np

from moptipy.api.encoding import Encoding
from moptipy.examples.jssp.gantt import Gantt
from moptipy.examples.jssp.instance import Instance
from moptipy.utils.nputils import int_range_to_dtype


@numba.njit(nogil=True, cache=True)
def _map(x: np.ndarray,
         machine_time: np.ndarray,
         job_time: np.ndarray,
         job_idx: np.ndarray,
         matrix: np.ndarray,
         times: np.ndarray) -> np.number:
    """
    Map an operation-based encoded array to a Gantt chart.

    :param np.ndarray x: the source array
    :param np.ndarray machine_time: a scratch array for machine times
    :param np.ndarray job_time: a scratch array for job times
    :param np.ndarray job_idx: a scratch array for job indices
    :param np.ndarray matrix: the instance matrix
    :param np.ndarray times: the output times array
    :return: the makespan
    :rtype: np.integer
    """
    machine_time.fill(0)
    job_time.fill(0)
    job_idx.fill(0)

    for job in x:
        idx = job_idx[job]
        job_idx[job] = idx + 2
        machine = matrix[job, idx]
        time = matrix[job, idx + 1]
        start = max(job_time[job], machine_time[machine])
        times[job, machine, 0] = start
        end = start + time
        times[job, machine, 1] = end
        machine_time[machine] = end
        job_time[job] = end

    return job_time.max()


class OperationBasedEncoding(Encoding):
    """
    An operation-based encoding for the Job Shop Scheduling Problem (JSSP).

    The operation-based encoding for the Job Shop Scheduling Problem (JSSP)
    maps permutations with repetitions to Gantt charts.
    In the operation-based encoding, the search space consists of permutations
    with repetitions of length `n*m`, where `n` is the number of jobs in the
    JSSP instance and `m` is the number of machines.
    In such a permutation with repetitions, each of the job ids from `0..n-1`
    occurs exactly `m` times.
    In the encoding, the permutations are processed from beginning to end and
    the jobs are assigned to machines in exactly the order in which they are
    encountered. If a job is encountered for the first time, we place its
    first operation onto the corresponding machine. If we encounter it for the
    second time, its second operation is placed on its corresponding machine,
    and so on.
    """

    def __init__(self, instance: Instance) -> None:
        """
        Instantiate the operation based encoding.

        :param moptipy.examples.jssp.Instance instance: the JSSP instance
        """
        if not isinstance(instance, Instance):
            raise ValueError("instance must be valid Instance, "
                             f"but is '{type(instance)}'.")
        dtype = int_range_to_dtype(instance.makespan_lower_bound,
                                   instance.makespan_upper_bound)

        self.__machine_time: Final[np.ndarray] = \
            np.zeros(instance.machines, dtype)

        self.__job_time: Final[np.ndarray] = \
            np.zeros(instance.jobs, dtype)

        self.__job_idx: Final[np.ndarray] = \
            np.zeros(instance.jobs, int_range_to_dtype(0, instance.jobs))

        self.__matrix: Final[np.ndarray] = instance.matrix

    def map(self, x: np.ndarray, y: Gantt) -> None:
        """
        Map an operation-based encoded array to a Gantt chart.

        :param np.array x: the array
        :param moptipy.examples.jssp.Gantt y: the Gantt chart
        """
        y.makespan = int(_map(x, self.__machine_time,
                              self.__job_time, self.__job_idx,
                              self.__matrix, y.times))

    def get_name(self) -> str:
        """
        Get the name of this encoding.

        :return: `"operation_based_encoding"`
        :rtype: str
        """
        return "operation_based_encoding"
