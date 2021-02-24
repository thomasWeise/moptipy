from importlib import resources
import numpy as np
from moptipy.utils import logging
from moptipy.api import Component
from moptipy.utils.logger import KeyValuesSection
from moptipy.utils.nputils import int_range_to_dtype
from typing import Final, List


class JSSPInstance(Component):
    """An instance of the Job Shop Scheduling Problem."""

    #: The number of machines in the instance.
    MACHINES: Final = "machines"
    #: The number of jobs in the instance.
    JOBS: Final = "jobs"
    #: The lower bound of the makespan of the instance.
    MAKESPAN_LOWER_BOUND: Final = "makespanLowerBound"

    def __init__(self,
                 name: str,
                 machines: int,
                 jobs: int,
                 matrix: np.ndarray,
                 makespan_lower_bound: int = None):
        """
        Create an instance of the Job Shop Scheduling Problem.
        :param name: the name of the instance
        :param machines: the number of machines
        :param jobs: the number of jobs
        :param matrix: the matrix with the data
        :param makespan_lower_bound: the lower bound of the makespan
        """

        self.name = logging.sanitize_name(name)
        """The name of this JSSP instance."""

        if name != self.name:
            ValueError("Name '" + name + "' is not a valid name.")

        if not isinstance(machines, int) or (machines < 1):
            ValueError("There must be at least one machine, but '"
                       + str(machines) + "' were specified in instance '"
                       + name + "'.")
        self.machines = machines
        """The number of machines in this JSSP instance."""

        if not isinstance(jobs, int) or (jobs < 1):
            ValueError("There must be at least one job, but '"
                       + str(jobs) + "' were specified in instance '"
                       + name + "'.")
        self.jobs = jobs
        """The number of jobs in this JSSP instance."""

        if not isinstance(matrix, np.ndarray):
            ValueError("The matrix must be an numpy.ndarray, but is a '"
                       + str(type(matrix)) + "' in instance '"
                       + name + "'.")

        if len(matrix.shape) != 2:
            ValueError("JSSP instance data matrix must have two dimensions, "
                       "but has '" + str(len(matrix.shape))
                       + "' in instance '" + name + "'.")

        if matrix.shape[0] != jobs:
            ValueError("Invalid shape '" + str(matrix.shape)
                       + "' of matrix: must have jobs=" + str(jobs)
                       + " rows, but has " + str(matrix.shape[0])
                       + " in instance '" + name + "'.")

        if matrix.shape[1] != 2 * machines:
            ValueError("Invalid shape '" + str(matrix.shape)
                       + "' of matrix: must have 2*machines="
                       + str(2 * machines) + " columns, but has "
                       + str(matrix.shape[0]) + " in instance '"
                       + name + "'.")
        if not np.issubdtype(matrix.dtype, np.integer):
            ValueError("Matrix must have an integer type, but is of type '"
                       + str(matrix.dtype) + "' in instance '"
                       + name + "'.")
        self.matrix = matrix
        """
        The matrix with the operations of the jobs and their durations.
        This matrix holds one row for each job.
        In each row, it stores tuples of (machine, duration) in a consecutive
        sequence, i.e., 2*machine numbers.
        """

        i64 = np.dtype(np.int64)
        usedmachines = np.zeros(machines, np.dtype(np.bool_))
        jobtimes = np.zeros(jobs, i64)
        machinetimes = np.zeros(machines, i64)
        machine1 = np.zeros(machines, i64)
        machine2 = np.zeros(machines, i64)

        jobidx = 0
        for row in matrix:
            usedmachines.fill(False)
            j = 0
            jobtime = 0
            for i in range(machines):
                machine = row[j]
                time = row[j + 1]
                if usedmachines[i]:
                    ValueError("Machine " + str(machine)
                               + " occurs more than once for instance '"
                               + name + "'.")
                usedmachines[i] = True
                if time < 0:
                    ValueError("Invalid time '" + str(time)
                               + "' for instance '"
                               + name + "'.")
                machinetimes[machine] += time
                machine1[machine] = min(machine1[machine], jobtime)
                jobtime += time
                j += 2

            jobtimes[jobidx] = jobtime
            j = 0
            for i in range(machines):
                machine = row[j]
                time = row[j + 1]
                machine2[machine] = min(machine2[machine], jobtime)
                jobtime -= time
                j += 2

            if not all(usedmachines):
                ValueError("Some machines not used in a job in instance '"
                           + name + "'.")
            jobidx += 1

        ms_bound = max(int(jobtimes.max()),
                       int((machine1 + machine2 + machinetimes).max()))
        if ms_bound <= 0:
            ValueError("Computed bound must not be <= 0, but is '"
                       + str(ms_bound) + "'.")

        if makespan_lower_bound is None:
            makespan_lower_bound = ms_bound
        else:
            if (not isinstance(makespan_lower_bound, int)) or \
                    (makespan_lower_bound <= 0):
                ValueError("If specified, makespan_lower_bound must be "
                           "positive integer, but is "
                           + str(makespan_lower_bound) + " in instance '"
                           + name + "'.")
            if makespan_lower_bound < ms_bound:
                ValueError("If specified, makespan_lower_bound must be >= "
                           + str(ms_bound) + ", but is "
                           + str(makespan_lower_bound) + " in instance '"
                           + name + "'.")

        self.makespan_lower_bound = makespan_lower_bound
        """The lower bound of the makespan for the JSSP instance."""

    def get_name(self):
        return self.name

    def log_parameters_to(self, logger: KeyValuesSection):
        super().log_parameters_to(logger)
        logger.key_value(JSSPInstance.MACHINES, self.machines)
        logger.key_value(JSSPInstance.JOBS, self.jobs)
        if not (self.makespan_lower_bound is None):
            logger.key_value(JSSPInstance.MAKESPAN_LOWER_BOUND,
                             self.makespan_lower_bound)

    @staticmethod
    def from_text(name: str,
                  rows: List[str]) -> 'JSSPInstance':
        """
        Convert a name and a set of rows of text to an JSSP instance.
        :param str name: the name of the instance
        :param List[str] rows: the rows
        :return: the JSSP Instance
        :rtype: JSSPInstance
        """
        if len(rows) < 3:
            raise ValueError("Must have at least 3 rows, but found "
                             + str(rows))
        description = rows[0]
        jobs_machines = rows[1]

        basetype = np.dtype(np.uint64)
        matrix = np.asanyarray([np.fromstring(row, dtype=basetype, sep=" ")
                                for row in rows[2:]])
        if not np.issubdtype(matrix.dtype, np.integer):
            ValueError("Error when converting array to matrix, got type '"
                       + str(matrix.dtype) + "'.")

        min_value = int(matrix.min())
        if min_value < 0:
            ValueError("JSSP matrix can only contain values >= 0, but found"
                       + str(min_value) + ".")

        max_value = int(matrix.max())
        if max_value <= min_value:
            ValueError("JSSP matrix must contain value larger than minimum "
                       + str(min_value) + ", but maximum is "
                       + str(max_value) + ".")

        dtype = int_range_to_dtype(min_value=min_value, max_value=max_value)
        if dtype != matrix.dtype:
            matrix = matrix.astype(dtype)

        jobs_machines = jobs_machines.strip().split(" ")
        jobs = int(jobs_machines[0])
        machines = int(jobs_machines[len(jobs_machines) - 1])

        makespan_lower_bound = None
        i = description.find("lower bound:")
        if i >= 0:
            i += 12
            j = description.find(";", i)
            if j < i:
                j = len(description)
            makespan_lower_bound = int(description[i:j].strip())

        return JSSPInstance(name=name,
                            jobs=jobs,
                            machines=machines,
                            matrix=matrix,
                            makespan_lower_bound=makespan_lower_bound)

    @staticmethod
    def from_stream(name: str, stream) -> 'JSSPInstance':
        """
        Load an instance from a text stream
        :param str name: the name of the instance to be loaded
        :param stream: the text stream
        :return: the instance
        :rtype: JSSPInstance
        """
        state = 0
        rows = None
        for line in stream:
            line = str(line).strip()
            if len(line) <= 0:
                continue
            if state == 0:
                if line.startswith("+++"):
                    state = 1
                continue
            if state == 1:
                if line.startswith("instance "):
                    inst = line[9:].strip()
                    if inst == name:
                        state = 2
                    else:
                        state = 4
            elif state == 2:
                if line.startswith("+++"):
                    state = 3
                    rows = []
                    continue
                raise ValueError("Unexpected string '" + line + "'.")
            elif state == 3:
                if line.startswith("+++"):
                    return JSSPInstance.from_text(name=name,
                                                  rows=rows)
                rows.append(line)
            elif state == 4:
                if line.startswith("+++"):
                    state = 5
                    continue
                raise ValueError("Unexpected string '" + line + "'.")
            elif state == 5:
                if line.startswith("+++"):
                    state = 1

        raise ValueError("Could not find instance '" + name + "'.")

    @staticmethod
    def from_resource(name: str,
                      package: str = str(__package__)) -> 'JSSPInstance':
        """
        Load one of the JSSP instances provided as part of this example
        package.
        :param str name: the instance name
        :param str package: the package
        :return: the instance
        :rtype: JSSPInstance
        """
        with resources.open_text(package=package,
                                 resource="demo.txt" if (name == "demo")
                                 else "instances.txt") as stream:
            return JSSPInstance.from_stream(name=name, stream=stream)
