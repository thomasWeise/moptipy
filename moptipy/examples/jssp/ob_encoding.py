"""
An implementation of the operation-based encoding for JSSPs.

1. Mitsuo Gen, Yasuhiro Tsujimura, and Erika Kubota. Solving Job-Shop
   Scheduling Problems by Genetic Algorithm. In *Humans, Information and
   Technology: Proceedings of the 1994 IEEE International Conference on
   Systems, Man and Cybernetics,* October 2-5, 1994, San Antonio, TX, USA,
   volume 2. Piscataway, NJ, USA: IEEE. ISBN: 0-7803-2129-4.
   doi: https://doi.org/10.1109/ICSMC.1994.400072.
   http://read.pudn.com/downloads151/doc/658565/00400072.pdf.
2. Christian Bierwirth. A Generalized Permutation Approach to Job Shop
   Scheduling with Genetic Algorithms. *Operations-Research-Spektrum
   (OR Spectrum)*, 17(2-3):87-92, June 1995.
   doi: https://doi.org/10.1007/BF01719250.
   https://www.researchgate.net/publication/240263036.
3. Christian Bierwirth, Dirk C. Mattfeld, and Herbert Kopfer. On Permutation
   Representations for Scheduling Problems. In Hans-Michael Voigt, Werner
   Ebeling, Ingo Rechenberg, and Hans-Paul Schwefel, editors, *Proceedings of
   the 4th International Conference on Parallel Problem Solving from Nature
   (PPSN IV),* September 22-24, 1996, Berlin, Germany, pages 310-318.
   Volume 1141/1996 of Lecture Notes in Computer Science (LNCS), Berlin,
   Germany: Springer-Verlag GmbH. ISBN: 3-540-61723-X.
   doi: https://doi.org/10.1007/3-540-61723-X_995.
   https://www.researchgate.net/publication/2753293.
4. Guoyong Shi, Hitoshi Iima, and Nobuo Sannomiya. New Encoding Scheme for
   Solving Job Shop Problems by Genetic Algorithm. In *Proceedings of the 35th
   IEEE Conference on Decision and Control (CDC'96),* December 11-13, 1996,
   Kobe, Japan, volume 4, pages 4395-4400. Piscataway, NJ, USA: IEEE.
   ISBN: 0-7803-3590-2. doi: https://doi.org/10.1109/CDC.1996.577484.
   https://www.researchgate.net/publication/224238934.
"""
from typing import Final

import numba  # type: ignore
import numpy as np

from moptipy.api.encoding import Encoding
from moptipy.examples.jssp.instance import Instance
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.nputils import (
    KEY_NUMPY_TYPE,
    int_range_to_dtype,
    numpy_type_to_str,
)
from moptipy.utils.types import type_error

#: the numpy data type for machine indices
KEY_NUMPY_TYPE_MACHINE_IDX: Final[str] = f"{KEY_NUMPY_TYPE}MachineIdx"
#: the numpy data type for job indices
KEY_NUMPY_TYPE_JOB_IDX: Final[str] = f"{KEY_NUMPY_TYPE}JobIdx"
#: the numpy data type for job times
KEY_NUMPY_TYPE_JOB_TIME: Final[str] = f"{KEY_NUMPY_TYPE}JobTime"


# start book
@numba.njit(nogil=True, cache=True)
def decode(x: np.ndarray, machine_idx: np.ndarray,
           job_time: np.ndarray, job_idx: np.ndarray,
           instance: np.ndarray, y: np.ndarray) -> None:
    """
    Map an operation-based encoded array to a Gantt chart.

    :param x: the source array, i.e., multi-permutation
    :param machine_idx: array of length `m` for machine indices
    :param job_time: array of length `n` for job times
    :param job_idx: length `n` array of current job operations
    :param instance: the instance data matrix
    :param y: the output array, i.e., the Gantt chart
    """
    machine_idx.fill(-1)  # all machines start by having done no jobs
    job_time.fill(0)  # each job has initially consumed 0 time units
    job_idx.fill(0)  # each job starts at its first operation

    for job in x:  # iterate over multi-permutation
        idx = job_idx[job]  # get the current operation of the job
        job_idx[job] = idx + 1  # and step it to the next operation
        machine = instance[job, idx, 0]  # get the machine id
        start = job_time[job]  # end time of previous operation of job
        mi = machine_idx[machine]  # get jobs finished on machine - 1
        if mi >= 0:  # we already have one job done?
            start = max(start, y[machine, mi, 2])  # earliest start
        mi += 1  # step the machine index
        machine_idx[machine] = mi  # step the machine index
        end = start + instance[job, idx, 1]  # compute end time
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

        :param instance: the JSSP instance
        """
        if not isinstance(instance, Instance):
            raise type_error(instance, "instance", Instance)
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

    def decode(self, x: np.ndarray, y: np.ndarray) -> None:  # +book
        """
        Map an operation-based encoded array to a Gantt chart.

        :param x: the array
        :param y: the Gantt chart
        """
        decode(x, self.__machine_idx, self.__job_time,  # +book
               self.__job_idx, self.__instance, y)  # +book

    def __str__(self) -> str:
        """
        Get the name of this encoding.

        :return: `"operation_based_encoding"`
        :rtype: str
        """
        return "operation_based_encoding"

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log all parameters of this component as key-value pairs.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        logger.key_value(KEY_NUMPY_TYPE_MACHINE_IDX,
                         numpy_type_to_str(self.__machine_idx.dtype))
        logger.key_value(KEY_NUMPY_TYPE_JOB_IDX,
                         numpy_type_to_str(self.__job_idx.dtype))
        logger.key_value(KEY_NUMPY_TYPE_JOB_TIME,
                         numpy_type_to_str(self.__job_time.dtype))
