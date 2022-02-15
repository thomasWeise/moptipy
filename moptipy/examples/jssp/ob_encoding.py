"""An implementation of the operation-based encoding for JSSPs."""
from typing import Final

import numba  # type: ignore
import numpy as np

from moptipy.api.encoding import Encoding
from moptipy.examples.jssp.instance import Instance
from moptipy.utils.logger import KeyValueSection
from moptipy.utils.nputils import int_range_to_dtype, KEY_NUMPY_TYPE

#: the numpy data type for machine indices
KEY_NUMPY_TYPE_MACHINE_IDX: Final[str] = f"{KEY_NUMPY_TYPE}MachineIdx"
#: the numpy data type for job indices
KEY_NUMPY_TYPE_JOB_IDX: Final[str] = f"{KEY_NUMPY_TYPE}JobIdx"
#: the numpy data type for job times
KEY_NUMPY_TYPE_JOB_TIME: Final[str] = f"{KEY_NUMPY_TYPE}JobTime"


# start book
@numba.njit(nogil=True, cache=True)
def decode(x: np.ndarray,
           machine_idx: np.ndarray,
           job_time: np.ndarray,
           job_idx: np.ndarray,
           matrix: np.ndarray,
           y: np.ndarray) -> None:
    """
    Map an operation-based encoded array to a Gantt chart.

    :param np.ndarray x: the source array, i.e., multi-permutation
    :param np.ndarray machine_idx: array of length `m` for machine indices
    :param np.ndarray job_time: array of length `n` for job times
    :param np.ndarray job_idx: length `n` array of current job operations
    :param np.ndarray matrix: the instance data matrix
    :param np.ndarray y: the output array: `times` of the Gantt chart
    """
    machine_idx.fill(-1)  # all machines start by having done no jobs
    job_time.fill(0)  # each job has initially consumed 0 time units
    job_idx.fill(0)  # each job starts at its first operation

    for job in x:  # iterate over multi-permutation
        idx = job_idx[job]  # get the current operation of the job
        job_idx[job] = idx + 1  # and step it to the next operation
        machine = matrix[job, idx, 0]  # get the machine id
        start = job_time[job]  # end time of previous operation of job
        mi = machine_idx[machine]  # get jobs finished on machine - 1
        if mi >= 0:  # we already have one job done?
            start = max(start, y[machine, mi, 2])  # earliest start
        mi += 1  # step the machine index
        machine_idx[machine] = mi  # step the machine index
        end = start + matrix[job, idx, 1]  # compute end time
        y[machine, mi, 0] = job  # store job index
        y[machine, mi, 1] = start  # store start of job's operation
        y[machine, mi, 2] = end  # store end of job's operation
        job_time[job] = end  # time next operation of job can start
    # end book


# start book
class OperationBasedEncoding(Encoding):
    # reusable variables __machine_time, __job_time, and __job_idx are
    # allocated in __init__; __matrix refers to instance data matrix
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
        self.__machine_idx: Final[np.ndarray] = \
            np.zeros(instance.machines,
                     int_range_to_dtype(-1, instance.jobs - 1))
        self.__job_time: Final[np.ndarray] = \
            np.zeros(instance.jobs,
                     int_range_to_dtype(0, instance.makespan_upper_bound))
        self.__job_idx: Final[np.ndarray] = \
            np.zeros(instance.jobs,
                     int_range_to_dtype(0, instance.machines - 1))
        self.__instance: Final[Instance] = instance

    def map(self, x: np.ndarray, y: np.ndarray) -> None:  # +book
        """
        Map an operation-based encoded array to a Gantt chart.

        :param np.array x: the array
        :param moptipy.examples.jssp.Gantt y: the Gantt chart
        """
        decode(x, self.__machine_idx, self.__job_time,  # +book
               self.__job_idx, self.__instance, y)  # +book

    def get_name(self) -> str:
        """
        Get the name of this encoding.

        :return: `"operation_based_encoding"`
        :rtype: str
        """
        return "operation_based_encoding"

    def log_parameters_to(self, logger: KeyValueSection) -> None:
        """
        Log all parameters of this component as key-value pairs.

        :param moptipy.utils.KeyValueSection logger: the logger
        """
        super().log_parameters_to(logger)
        logger.key_value(KEY_NUMPY_TYPE_MACHINE_IDX,
                         self.__machine_idx.dtype.char)
        logger.key_value(KEY_NUMPY_TYPE_JOB_IDX, self.__job_idx.dtype.char)
        logger.key_value(KEY_NUMPY_TYPE_JOB_TIME, self.__job_time.dtype.char)
