"""An implementation of the operation-based encoding for JSSPs."""
from typing import Final

import numba  # type: ignore
import numpy as np

from moptipy.api.encoding import Encoding
from moptipy.examples.jssp.gantt import Gantt
from moptipy.examples.jssp.instance import Instance
from moptipy.utils.nputils import int_range_to_dtype


# start book
@numba.njit(nogil=True, cache=True)
def decode(x: np.ndarray,
           machine_time: np.ndarray,
           job_time: np.ndarray,
           job_idx: np.ndarray,
           matrix: np.ndarray,
           y: np.ndarray) -> None:
    """
    Map an operation-based encoded array to a Gantt chart.

    :param np.ndarray x: the source array, i.e., multi-permutation
    :param np.ndarray machine_time: array of length `m` for machine times
    :param np.ndarray job_time: array of length `n` for job times
    :param np.ndarray job_idx: length `n` array of current job operations
    :param np.ndarray matrix: the instance data matrix
    :param np.ndarray y: the output array: `times` of the Gantt chart
    """
    machine_time.fill(0)  # all machines start at time 0
    job_time.fill(0)  # each job has initially consumed 0 time units
    job_idx.fill(0)  # each job starts at its first operation

    for job in x:  # iterate over multi-permutation
        idx = job_idx[job]  # get the current operation of the job
        job_idx[job] = idx + 2  # and step it to the next operation
        machine = matrix[job, idx]  # get the machine of the operation
        time = matrix[job, idx + 1]  # and the time requirement
        start = max(job_time[job], machine_time[machine])  # earliest
        y[job, machine, 0] = start  # store start time in Gantt chart
        end = start + time  # compute end time
        y[job, machine, 1] = end  # store end time in Gantt chart
        machine_time[machine] = end  # time when next job can start
        job_time[job] = end  # time next operation of job can start


class OperationBasedEncoding(Encoding):
    # reusable variables __machine_time, __job_time, and __job_idx are
    # allocated in __init__; __matrix points to instance data I.matrix
    # end book
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

    # start book
    def map(self, x: np.ndarray, y: Gantt) -> None:
        """
        Map an operation-based encoded array to a Gantt chart.

        :param np.array x: the array
        :param moptipy.examples.jssp.Gantt y: the Gantt chart
        """
        decode(x, self.__machine_time, self.__job_time, self.__job_idx,
               self.__matrix, y.times)
        # end book

    def get_name(self) -> str:
        """
        Get the name of this encoding.

        :return: `"operation_based_encoding"`
        :rtype: str
        """
        return "operation_based_encoding"
