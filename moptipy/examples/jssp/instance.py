"""Here we provide a representation for JSSP instances."""
from importlib import resources
from typing import Final, List, Tuple, Optional

import numpy as np

from moptipy.api import Component
from moptipy.utils import logging
from moptipy.utils import nputils
from moptipy.utils.logger import KeyValueSection
from moptipy.utils.nputils import int_range_to_dtype

#: the recommended scope under which instance data should be stored
SCOPE_INSTANCE: Final = "inst"
#: The number of machines in the instance.
MACHINES: Final = "machines"
#: The number of jobs in the instance.
JOBS: Final = "jobs"
#: The lower bound of the makespan of the instance.
MAKESPAN_LOWER_BOUND: Final = "makespanLowerBound"
#: The upper bound of the makespan of the instance.
MAKESPAN_UPPER_BOUND: Final = "makespanUpperBound"


# start lb
def compute_makespan_lower_bound(machines: int,
                                 jobs: int,
                                 matrix: np.ndarray) -> int:
    """
    Compute the lower bound for the makespan of a JSSP instance.

    :param int machines: the number of machines
    :param int jobs: the number of jobs
    :param np.ndarray matrix: the matrix with the instance data
    :returns: the lower bound for the makespan
    :rtype: int
    """
    # get the lower bound of the makespan with the algorithm by Taillard
    usedmachines = np.zeros(machines, np.bool_)  # -lb
    jobtimes = np.zeros(jobs, np.int64)  # allocate array for job times
    machinetimes = np.zeros(machines, np.int64)  # machine times array
    machine_start_idle = nputils.np_ints_max(machines, nputils.DEFAULT_INT)
    machine_end_idle = nputils.np_ints_max(machines, nputils.DEFAULT_INT)

    for jobidx in range(jobs):  # iterate over all jobs
        row = matrix[jobidx]  # get the data for the job
        usedmachines.fill(False)  # no machine has been used  # -lb
        j: int = 0  # the index into the data
        jobtime: int = 0  # the job time sum
        for i in range(machines):  # iterate over all operations
            machine: int = row[j]  # get machine for operation
            time: int = row[j + 1]  # get time for operation
            if usedmachines[i]:  # machine already used??? -> error  # -lb
                raise ValueError(  # -lb
                    f"Machine {machine} occurs more than once.")  # -lb
            usedmachines[i] = True  # mark machine as used  # -lb
            if time < 0:  # time can _never_ be negative -> error  # -lb
                raise ValueError(f"Invalid time '{time}'.")  # -lb
            machinetimes[machine] += time  # add up operation times
            machine_start_idle[machine] = min(  # update start idle time
                machine_start_idle[machine], jobtime)  # with job time
            jobtime += time  # update job time by adding operation time
            j += 2  # step operation index

        jobtimes[jobidx] = jobtime  # store job time
        jobremaining = jobtime  # iterate backwards to get end idle times
        for i in range(machines):  # second iteration round
            j -= 2  # step operation index downwards
            machine = row[j]  # get machine for operation
            time = row[j + 1]  # get time for operation
            machine_end_idle[machine] = min(  # update machine end idle
                machine_end_idle[machine],  # time by computing the time
                jobtime - jobremaining)  # the job needs after operation
            jobremaining -= time  # and update the remaining job time

        if not all(usedmachines):  # all machines have been used?  # -lb
            raise ValueError("Some machines not used in a job.")  # -lb

    # get the maximum of the per-machine sums of the idle and work times
    machines_bound = (machine_start_idle + machine_end_idle
                      + machinetimes).max()
    if machines_bound <= 0:  # -lb
        raise ValueError("Computed machine bound cannot be <= , "  # -lb
                         f"but is {machines_bound}.")  # -lb
    # get  the longest time any job needs in total
    jobs_bound = jobtimes.max()
    if jobs_bound <= 0:  # -lb
        raise ValueError(  # -lb
            f"Computed jobs bound cannot be <= , but is {jobs_bound}.")  # -lb

    return int(max(machines_bound, jobs_bound))  # return the maximum
# end lb


# start book
class Instance(Component):
    """An instance of the Job Shop Scheduling Problem."""

    def __init__(self, name: str, machines: int, jobs: int,
                 matrix: np.ndarray,
                 makespan_lower_bound: Optional[int] = None) -> None:
        """
        Create an instance of the Job Shop Scheduling Problem.

        :param str name: the name of the instance
        :param int machines: the number of machines
        :param int jobs: the number of jobs
        :param np.ndarray matrix: the matrix with the data
        :param Optional[int] makespan_lower_bound: the lower bound of the
            makespan, which may be the known global optimum if the
            instance has been solved to optimality or any other
            approximation. If `None` is provided, a lower bound will be
            computed.
        """
        # end book
        #: The name of this JSSP instance.
        self.name: Final[str] = logging.sanitize_name(name)  # +book

        if name != self.name:
            raise ValueError(f"Name '{name}' is not a valid name.")

        if (not isinstance(machines, int)) or (machines < 1):
            raise ValueError("There must be at least one machine, "
                             f"but '{machines}' were specified in "
                             f"instance '{name}'.")
        #: The number of machines in this JSSP instance.
        self.machines: Final[int] = machines  # +book

        if not isinstance(jobs, int) or (jobs < 1):
            raise ValueError("There must be at least one job, "
                             f"but '{jobs}' were specified in "
                             f"instance '{name}'.")
        #: The number of jobs in this JSSP instance.
        self.jobs: Final[int] = jobs  # +book

        if not isinstance(matrix, np.ndarray):
            raise TypeError("The matrix must be an numpy.ndarray, but is a "
                            f"'{type(matrix)}' in instance '{name}'.")

        if len(matrix.shape) != 2:
            raise ValueError(
                "JSSP instance data matrix must have two dimensions, "
                f"but has {len(matrix.shape)} in instance '{name}'.")

        if matrix.shape[0] != jobs:
            raise ValueError(
                f"Invalid shape '{matrix.shape}' of matrix: must have "
                f"jobs={jobs} rows, but has {matrix.shape[0]} in "
                f"instance '{name}'.")

        if matrix.shape[1] != 2 * machines:
            raise ValueError(
                f"Invalid shape '{matrix.shape}' of matrix: must have "
                f"2*machines={2 * machines} columns, but has "
                f"{matrix.shape[0]} in instance '{name}'.")
        if not np.issubdtype(matrix.dtype, np.integer):
            raise ValueError(
                "Matrix must have an integer type, but is of type "
                f"'{matrix.dtype}' in instance '{name}'.")
        #: The matrix with the operations of the jobs and their durations. \
        #: This matrix holds one row for each job. \
        #: In each row, it stores tuples of (machine, duration) in a \
        #: consecutive sequence, i.e., 2*machine numbers.
        self.matrix: Final[np.ndarray] = matrix  # +book

        # ... some computations ...  # +book
        ms_lower_bound = compute_makespan_lower_bound(machines, jobs, matrix)
        ms_upper_bound = int(matrix[:, 1::2].sum())  # sum of all job times
        if ms_upper_bound < ms_lower_bound:
            raise ValueError(
                f"Computed makespan upper bound {ms_upper_bound} must not "
                f"be less than computed lower bound {ms_lower_bound}.")

        if makespan_lower_bound is None:
            makespan_lower_bound = ms_lower_bound
        else:
            if not isinstance(makespan_lower_bound, int):
                raise TypeError("Makespan lower bound, if provided, must int,"
                                f" but is {type(makespan_lower_bound)} in "
                                f"instance '{name}'.")
            if makespan_lower_bound <= 0:
                raise ValueError("If specified, makespan_lower_bound must be "
                                 f"positive, but is {makespan_lower_bound} "
                                 f"in instance '{name}'.")
            if makespan_lower_bound < ms_lower_bound:
                raise ValueError(
                    "If specified, makespan_lower_bound must be >= "
                    f"{ms_lower_bound}, but is {makespan_lower_bound} in "
                    f"instance '{name}'.")
            if makespan_lower_bound > ms_upper_bound:
                raise ValueError(
                    "If specified, makespan_lower_bound must be <= "
                    f"{ms_upper_bound}, but is {makespan_lower_bound} in "
                    f"instance '{name}'.")

        #: The lower bound of the makespan for the JSSP instance.
        self.makespan_lower_bound: Final[int] = makespan_lower_bound  # +book

        #: The upper bound of the makespan for the JSSP instance.
        self.makespan_upper_bound: Final[int] = ms_upper_bound

    def get_name(self) -> str:
        """
        Get the name of this JSSP instance.

        :return: the name
        :rtype: str
        """
        return self.name

    def log_parameters_to(self, logger: KeyValueSection) -> None:
        """
        Log the parameters describing this JSSP instance to the logger.

        :param moptipy.utils.KeyValueSection logger: the logger
        """
        super().log_parameters_to(logger)
        logger.key_value(MACHINES, self.machines)
        logger.key_value(JOBS, self.jobs)
        logger.key_value(MAKESPAN_LOWER_BOUND,
                         self.makespan_lower_bound)
        logger.key_value(MAKESPAN_UPPER_BOUND,
                         self.makespan_upper_bound)

    @staticmethod
    def from_text(name: str, rows: List[str]) -> 'Instance':
        """
        Convert a name and a set of rows of text to an JSSP instance.

        :param str name: the name of the instance
        :param List[str] rows: the rows
        :return: the JSSP Instance
        :rtype: Instance
        """
        if not isinstance(rows, list):
            raise TypeError(
                f"rows must be list of str, but are {type(rows)}.")
        if len(rows) < 3:
            raise ValueError(
                f"Must have at least 3 rows, but found {rows}.")
        description = rows[0]
        if not isinstance(description, str):
            raise TypeError("rows must be list of str, "
                            f"but are List[{type(description)}].")
        jobs_machines_txt = rows[1]

        basetype = np.dtype(np.uint64)
        matrix = np.asanyarray([np.fromstring(row, dtype=basetype, sep=" ")
                                for row in rows[2:]])
        if not np.issubdtype(matrix.dtype, np.integer):
            raise ValueError("Error when converting array to matrix, "
                             f"got type '{matrix.dtype}'.")

        min_value = int(matrix.min())
        if min_value < 0:
            raise ValueError("JSSP matrix can only contain values >= 0, "
                             f"but found {min_value}.")

        max_value = int(matrix.max())
        if max_value <= min_value:
            raise ValueError(
                "JSSP matrix must contain value larger than minimum "
                f"{min_value}, but maximum is {max_value}.")

        dtype = int_range_to_dtype(min_value=min_value, max_value=max_value)
        if dtype != matrix.dtype:
            matrix = matrix.astype(dtype)

        jobs_machines = jobs_machines_txt.strip().split(" ")
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

        return Instance(name=name,
                        jobs=jobs,
                        machines=machines,
                        matrix=matrix,
                        makespan_lower_bound=makespan_lower_bound)

    @staticmethod
    def from_stream(name: str, stream) -> 'Instance':
        """
        Load an instance from a text stream.

        :param str name: the name of the instance to be loaded
        :param stream: the text stream
        :return: the instance
        :rtype: Instance
        """
        state = 0
        rows: Optional[List[str]] = None
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
                raise ValueError(f"Unexpected string '{line}'.")
            elif state == 3:
                if line.startswith("+++"):
                    return Instance.from_text(name=name,
                                              rows=rows)
                rows.append(line)
            elif state == 4:
                if line.startswith("+++"):
                    state = 5
                    continue
                raise ValueError(f"Unexpected string '{line}'.")
            elif state == 5:
                if line.startswith("+++"):
                    state = 1

        raise ValueError(f"Could not find instance '{name}'.")

    @staticmethod
    def from_resource(name: str) -> 'Instance':
        """
        Load the JSSP instances `name` provided as part of moptipy.

        :param str name: the instance name
        :return: the instance
        :rtype: Instance

        >>> jssp = Instance.from_resource("demo")
        >>> print(jssp.jobs)
        4
        >>> print(jssp.machines)
        5
        >>> print(jssp.get_name())
        demo
        """
        with resources.open_text(package=str(__package__),
                                 resource="demo.txt" if (name == "demo")
                                 else "instances.txt") as stream:
            return Instance.from_stream(name=name, stream=stream)

    @staticmethod
    def list_resources() -> Tuple[str, ...]:
        """
        A tuple with all the JSSP instances provided in the moptipy resources.

        :return: a tuple with all instance names that are valid parameters
            to :meth:`Instance.from_resource`

        >>> print(Instance.list_resources()[0:3])
        ('abz5', 'abz6', 'abz7')
        """
        return 'abz5', 'abz6', 'abz7', 'abz8', 'abz9', \
               'demo', \
               'dmu01', 'dmu02', 'dmu03', 'dmu04', 'dmu05', 'dmu06', 'dmu07', \
               'dmu08', 'dmu09', 'dmu10', 'dmu11', 'dmu12', 'dmu13', 'dmu14', \
               'dmu15', 'dmu16', 'dmu17', 'dmu18', 'dmu19', 'dmu20', 'dmu21', \
               'dmu22', 'dmu23', 'dmu24', 'dmu25', 'dmu26', 'dmu27', 'dmu28', \
               'dmu29', 'dmu30', 'dmu31', 'dmu32', 'dmu33', 'dmu34', 'dmu35', \
               'dmu36', 'dmu37', 'dmu38', 'dmu39', 'dmu40', 'dmu41', 'dmu42', \
               'dmu43', 'dmu44', 'dmu45', 'dmu46', 'dmu47', 'dmu48', 'dmu49', \
               'dmu50', 'dmu51', 'dmu52', 'dmu53', 'dmu54', 'dmu55', 'dmu56', \
               'dmu57', 'dmu58', 'dmu59', 'dmu60', 'dmu61', 'dmu62', 'dmu63', \
               'dmu64', 'dmu65', 'dmu66', 'dmu67', 'dmu68', 'dmu69', 'dmu70', \
               'dmu71', 'dmu72', 'dmu73', 'dmu74', 'dmu75', 'dmu76', 'dmu77', \
               'dmu78', 'dmu79', 'dmu80', \
               'ft06', 'ft10', 'ft20', \
               'la01', 'la02', 'la03', 'la04', 'la05', 'la06', 'la07', \
               'la08', 'la09', 'la10', 'la11', 'la12', 'la13', 'la14', \
               'la15', 'la16', 'la17', 'la18', 'la19', 'la20', 'la21', \
               'la22', 'la23', 'la24', 'la25', 'la26', 'la27', 'la28', \
               'la29', 'la30', 'la31', 'la32', 'la33', 'la34', 'la35', \
               'la36', 'la37', 'la38', 'la39', 'la40', \
               'orb01', 'orb02', 'orb03', 'orb04', 'orb05', 'orb06', 'orb07', \
               'orb08', 'orb09', 'orb10', \
               'swv01', 'swv02', 'swv03', 'swv04', 'swv05', 'swv06', 'swv07', \
               'swv08', 'swv09', 'swv10', 'swv11', 'swv12', 'swv13', 'swv14', \
               'swv15', 'swv16', 'swv17', 'swv18', 'swv19', 'swv20', \
               'ta01', 'ta02', 'ta03', 'ta04', 'ta05', 'ta06', 'ta07', \
               'ta08', 'ta09', 'ta10', 'ta11', 'ta12', 'ta13', 'ta14', \
               'ta15', 'ta16', 'ta17', 'ta18', 'ta19', 'ta20', 'ta21', \
               'ta22', 'ta23', 'ta24', 'ta25', 'ta26', 'ta27', 'ta28', \
               'ta29', 'ta30', 'ta31', 'ta32', 'ta33', 'ta34', 'ta35', \
               'ta36', 'ta37', 'ta38', 'ta39', 'ta40', 'ta41', 'ta42', \
               'ta43', 'ta44', 'ta45', 'ta46', 'ta47', 'ta48', 'ta49', \
               'ta50', 'ta51', 'ta52', 'ta53', 'ta54', 'ta55', 'ta56', \
               'ta57', 'ta58', 'ta59', 'ta60', 'ta61', 'ta62', 'ta63', \
               'ta64', 'ta65', 'ta66', 'ta67', 'ta68', 'ta69', 'ta70', \
               'ta71', 'ta72', 'ta73', 'ta74', 'ta75', 'ta76', 'ta77', \
               'ta78', 'ta79', 'ta80', \
               'yn1', 'yn2', 'yn3', 'yn4'
