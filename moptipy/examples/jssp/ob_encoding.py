from moptipy.api.encoding import Encoding
from moptipy.examples.jssp.gantt import Gantt
from moptipy.examples.jssp.instance import JSSPInstance
import numpy as np
from moptipy.utils.nputils import int_range_to_dtype


class OperationBasedEncoding(Encoding):
    """
    An operation-based encoding for the Job Shop Scheduling Problem (JSSP)
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

    def __init__(self, instance: JSSPInstance):
        """
        Instantiate the operation based encoding.
        :param JSSPInstance instance: the JSSP instance
        """
        if not isinstance(instance, JSSPInstance):
            raise ValueError("instance must be valid JSSPInstance, but is '"
                             + str(type(instance)) + "'.")
        dtype = int_range_to_dtype(instance.makespan_lower_bound,
                                   instance.makespan_upper_bound)
        self.__machine_time = np.zeros(instance.machines, dtype)
        self.__job_time = np.zeros(instance.jobs, dtype)
        self.__job_idx = np.zeros(instance.jobs,
                                  int_range_to_dtype(0, instance.jobs))
        self.__matrix = instance.matrix

    def map(self, x: np.ndarray, y: Gantt):
        machine_time = self.__machine_time
        machine_time.fill(0)
        job_time = self.__job_time
        job_time.fill(0)
        job_idx = self.__job_idx
        job_idx.fill(0)
        matrix = self.__matrix
        times = y.times

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

        y.makespan = job_time.max()

    def get_name(self) -> str:
        return "operation_based_encoding"
