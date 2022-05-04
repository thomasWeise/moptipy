"""
Here we provide a representation for JSSP instances.

Our problem instances are actually extensions of
:class:`~numpy.ndarray`. They present the basic instance data as a
matrix, where each row corresponds to a job. Each row has twice the
number of machines elements. In an alternating fashion, we store the
machines that the job needs to go to as well as the time that it needs
on the machines, one by one.
Additionally, the instance name, the number of machines, and the number
of jobs are provided as attributes (although the latter two can easily
be inferred from the shape of the matrix).
Nevertheless, this memory layout and encapsulation as numpy array are
the most efficient way to store the data I could come up with.
"""
from importlib import resources  # nosem
from typing import Final, List, Tuple, Optional

import numpy as np

import moptipy.utils.nputils as npu
from moptipy.api.component import Component
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.nputils import int_range_to_dtype
from moptipy.utils.strings import sanitize_name
from moptipy.utils.types import type_error

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

#: the internal final set of instances
_INSTANCES: Final[Tuple[str, ...]] = \
    ('abz5', 'abz6', 'abz7', 'abz8', 'abz9',
     'demo',
     'dmu01', 'dmu02', 'dmu03', 'dmu04', 'dmu05', 'dmu06', 'dmu07',
     'dmu08', 'dmu09', 'dmu10', 'dmu11', 'dmu12', 'dmu13', 'dmu14',
     'dmu15', 'dmu16', 'dmu17', 'dmu18', 'dmu19', 'dmu20', 'dmu21',
     'dmu22', 'dmu23', 'dmu24', 'dmu25', 'dmu26', 'dmu27', 'dmu28',
     'dmu29', 'dmu30', 'dmu31', 'dmu32', 'dmu33', 'dmu34', 'dmu35',
     'dmu36', 'dmu37', 'dmu38', 'dmu39', 'dmu40', 'dmu41', 'dmu42',
     'dmu43', 'dmu44', 'dmu45', 'dmu46', 'dmu47', 'dmu48', 'dmu49',
     'dmu50', 'dmu51', 'dmu52', 'dmu53', 'dmu54', 'dmu55', 'dmu56',
     'dmu57', 'dmu58', 'dmu59', 'dmu60', 'dmu61', 'dmu62', 'dmu63',
     'dmu64', 'dmu65', 'dmu66', 'dmu67', 'dmu68', 'dmu69', 'dmu70',
     'dmu71', 'dmu72', 'dmu73', 'dmu74', 'dmu75', 'dmu76', 'dmu77',
     'dmu78', 'dmu79', 'dmu80',
     'ft06', 'ft10', 'ft20',
     'la01', 'la02', 'la03', 'la04', 'la05', 'la06', 'la07',
     'la08', 'la09', 'la10', 'la11', 'la12', 'la13', 'la14',
     'la15', 'la16', 'la17', 'la18', 'la19', 'la20', 'la21',
     'la22', 'la23', 'la24', 'la25', 'la26', 'la27', 'la28',
     'la29', 'la30', 'la31', 'la32', 'la33', 'la34', 'la35',
     'la36', 'la37', 'la38', 'la39', 'la40',
     'orb01', 'orb02', 'orb03', 'orb04', 'orb05', 'orb06', 'orb07',
     'orb08', 'orb09', 'orb10',
     'swv01', 'swv02', 'swv03', 'swv04', 'swv05', 'swv06', 'swv07',
     'swv08', 'swv09', 'swv10', 'swv11', 'swv12', 'swv13', 'swv14',
     'swv15', 'swv16', 'swv17', 'swv18', 'swv19', 'swv20',
     'ta01', 'ta02', 'ta03', 'ta04', 'ta05', 'ta06', 'ta07',
     'ta08', 'ta09', 'ta10', 'ta11', 'ta12', 'ta13', 'ta14',
     'ta15', 'ta16', 'ta17', 'ta18', 'ta19', 'ta20', 'ta21',
     'ta22', 'ta23', 'ta24', 'ta25', 'ta26', 'ta27', 'ta28',
     'ta29', 'ta30', 'ta31', 'ta32', 'ta33', 'ta34', 'ta35',
     'ta36', 'ta37', 'ta38', 'ta39', 'ta40', 'ta41', 'ta42',
     'ta43', 'ta44', 'ta45', 'ta46', 'ta47', 'ta48', 'ta49',
     'ta50', 'ta51', 'ta52', 'ta53', 'ta54', 'ta55', 'ta56',
     'ta57', 'ta58', 'ta59', 'ta60', 'ta61', 'ta62', 'ta63',
     'ta64', 'ta65', 'ta66', 'ta67', 'ta68', 'ta69', 'ta70',
     'ta71', 'ta72', 'ta73', 'ta74', 'ta75', 'ta76', 'ta77',
     'ta78', 'ta79', 'ta80',
     'yn1', 'yn2', 'yn3', 'yn4')


# start lb
def compute_makespan_lower_bound(machines: int,
                                 jobs: int,
                                 matrix: np.ndarray) -> int:
    """
    Compute the lower bound for the makespan of a JSSP instance data.

    :param machines: the number of machines
    :param jobs: the number of jobs
    :param matrix: the matrix with the instance data
    :returns: the lower bound for the makespan
    """
    # get the lower bound of the makespan with the algorithm by Taillard
    usedmachines = np.zeros(machines, np.bool_)  # -lb
    jobtimes = np.zeros(jobs, npu.DEFAULT_INT)  # get array for job times
    machinetimes = np.zeros(machines, npu.DEFAULT_INT)  # machine times array
    machine_start_idle = npu.np_ints_max(machines, npu.DEFAULT_INT)
    machine_end_idle = npu.np_ints_max(machines, npu.DEFAULT_INT)

    for jobidx in range(jobs):  # iterate over all jobs
        usedmachines.fill(False)  # no machine has been used  # -lb
        jobtime: int = 0  # the job time sum
        for i in range(machines):  # iterate over all operations
            machine, time = matrix[jobidx, i]  # get operation data
            if usedmachines[i]:  # machine already used??? -> error  # -lb
                raise ValueError(  # -lb
                    f"Machine {machine} occurs more than once.")  # -lb
            usedmachines[i] = True  # mark machine as used  # -lb
            if time < 0:  # time can _never_ be negative -> error  # -lb
                raise ValueError(f"Invalid time '{time}'.")  # -lb
            machinetimes[machine] += time  # add up operation times
            machine_start_idle[machine] = min(  # update with...
                machine_start_idle[machine], jobtime)  # ...job time
            jobtime += time  # update job time by adding operation time

        jobtimes[jobidx] = jobtime  # store job time
        jobremaining = jobtime  # iterate backwards to get end idle times
        for i in range(machines - 1, -1, -1):  # second iteration round
            machine, time = matrix[jobidx, i]  # get machine for operation
            machine_end_idle[machine] = min(  # update by computing...
                machine_end_idle[machine],  # the time that the job...
                jobtime - jobremaining)  # needs _after_ operation
            jobremaining -= time  # and update the remaining job time

        if not all(usedmachines):  # all machines have been used?  # -lb
            raise ValueError("Some machines not used in a job.")  # -lb

    # get the maximum of the per-machine sums of the idle and work times
    machines_bound = (machine_start_idle + machine_end_idle
                      + machinetimes).max()
    if machines_bound <= 0:  # -lb
        raise ValueError("Computed machine bound cannot be <= , "  # -lb
                         f"but is {machines_bound}.")  # -lb
    # get the longest time any job needs in total
    jobs_bound = jobtimes.max()
    if jobs_bound <= 0:  # -lb
        raise ValueError(  # -lb
            f"Computed jobs bound cannot be <= , but is {jobs_bound}.")  # -lb

    return int(max(machines_bound, jobs_bound))  # return bigger one
# end lb


# start book
class Instance(Component, np.ndarray):
    """
    An instance of the Job Shop Scheduling Problem.

    Besides the metadata, this object is a three-dimensional np.ndarray
    where the columns stand for jobs and the rows represent the
    operations of the jobs. Each row*column contains two values (third
    dimension), namely the machine where the operation goes and the time
    it will consume at that machine: `I[job, operation, 0] = machine`,
    `I[job, operation, 1] = time` that the job spents on machine.
    """

    # the name of the instance
    name: str
    #: the number of jobs == self.shape[0]
    jobs: int
    #: the number of machines == self.shape[1]
    machines: int
    # ... some more properties and methods ...
    # end book
    #: the lower bound of the makespan of this JSSP instance
    makespan_lower_bound: int
    #: the upper bound of the makespan of this JSSP instance
    makespan_upper_bound: int

    def __new__(cls, name: str, machines: int, jobs: int,
                matrix: np.ndarray,
                makespan_lower_bound: Optional[int] = None) -> 'Instance':
        """
        Create an instance of the Job Shop Scheduling Problem.

        :param cls: the class
        :param name: the name of the instance
        :param machines: the number of machines
        :param jobs: the number of jobs
        :param matrix: the matrix with the data (will be copied)
        :param makespan_lower_bound: the lower bound of the makespan, which
            may be the known global optimum if the instance has been solved
            to optimality or any other approximation. If `None` is provided,
            a lower bound will be computed.
        """
        use_name: Final[str] = sanitize_name(name)
        if name != use_name:
            raise ValueError(f"Name '{name}' is not a valid name.")

        if (not isinstance(machines, int)) or (machines < 1):
            raise ValueError("There must be at least one machine, "
                             f"but '{machines}' were specified in "
                             f"instance '{name}'.")
        if not isinstance(jobs, int) or (jobs < 1):
            raise ValueError("There must be at least one job, "
                             f"but '{jobs}' were specified in "
                             f"instance '{name}'.")
        if not isinstance(matrix, np.ndarray):
            raise type_error(matrix, "matrix", np.ndarray)

        use_shape: Tuple[int, int, int] = (jobs, machines, 2)
        if matrix.shape[0] != jobs:
            raise ValueError(
                f"Invalid shape '{matrix.shape}' of matrix: must have "
                f"jobs={jobs} rows, but has {matrix.shape[0]} in "
                f"instance '{name}'.")
        if len(matrix.shape) == 3:
            if matrix.shape[1] != machines:
                raise ValueError(
                    f"Invalid shape '{matrix.shape}' of matrix: must have "
                    f"2*machines={machines} columns, but has "
                    f"{matrix.shape[1]} in instance '{name}'.")
            if matrix.shape[2] != 2:
                raise ValueError(
                    f"Invalid shape '{matrix.shape}' of matrix: must have "
                    f"2 cells per row/column tuple, but has "
                    f"{matrix.shape[2]} in instance '{name}'.")
        elif len(matrix.shape) == 2:
            if matrix.shape[1] != 2 * machines:
                raise ValueError(
                    f"Invalid shape '{matrix.shape}' of matrix: must have "
                    f"2*machines={2 * machines} columns, but has "
                    f"{matrix.shape[1]} in instance '{name}'.")
            matrix = matrix.reshape(use_shape)
        else:
            raise ValueError(
                "JSSP instance data matrix must have two or three"
                f"dimensions, but has {len(matrix.shape)} in instance "
                f"'{name}'.")
        if matrix.shape != use_shape:
            raise ValueError(
                f"matrix.shape is {matrix.shape}, not {use_shape}?")
        if not npu.is_np_int(matrix.dtype):
            raise ValueError(
                "Matrix must have an integer type, but is of type "
                f"'{matrix.dtype}' in instance '{name}'.")
        # ... some computations ...
        ms_lower_bound = compute_makespan_lower_bound(machines, jobs, matrix)
        ms_upper_bound = int(matrix[:, :, 1].sum())  # sum of all job times
        if ms_upper_bound < ms_lower_bound:
            raise ValueError(
                f"Computed makespan upper bound {ms_upper_bound} must not "
                f"be less than computed lower bound {ms_lower_bound}.")
        if makespan_lower_bound is None:
            makespan_lower_bound = ms_lower_bound
        else:
            if not isinstance(makespan_lower_bound, int):
                raise type_error(makespan_lower_bound,
                                 f"makespan lower bound of instance {name}",
                                 int)
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
        obj: Final[Instance] = super().__new__(
            Instance, use_shape, int_range_to_dtype(
                min_value=0, max_value=int(matrix.max())))
        np.copyto(obj, matrix, casting="safe")
        #: the name of the instance
        obj.name = use_name
        #: the number of jobs == self.shape[0]
        obj.jobs = jobs
        #: the number of machines == self.shape[1]
        obj.machines = machines
        #: the lower bound of the makespan of this JSSP instance
        obj.makespan_lower_bound = makespan_lower_bound
        #: the upper bound of the makespan of this JSSP instance
        obj.makespan_upper_bound = ms_upper_bound
        return obj

    def __str__(self) -> str:
        """
        Get the name of this JSSP instance.

        :return: the name
        """
        return self.name

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters describing this JSSP instance to the logger.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        logger.key_value(MACHINES, self.machines)
        logger.key_value(JOBS, self.jobs)
        logger.key_value(MAKESPAN_LOWER_BOUND, self.makespan_lower_bound)
        logger.key_value(MAKESPAN_UPPER_BOUND, self.makespan_upper_bound)
        logger.key_value(npu.KEY_NUMPY_TYPE, self.dtype.char)

    @staticmethod
    def from_text(name: str, rows: List[str]) -> 'Instance':
        """
        Convert a name and a set of rows of text to an JSSP instance.

        :param name: the name of the instance
        :param rows: the rows
        :return: the JSSP Instance
        """
        if not isinstance(rows, list):
            raise type_error(rows, "rows", List)
        if len(rows) < 3:
            raise ValueError(
                f"Must have at least 3 rows, but found {rows}.")
        description = rows[0]
        if not isinstance(description, str):
            raise type_error(description, "first element of rows", str)
        jobs_machines_txt = rows[1]

        matrix = np.asanyarray([np.fromstring(row, dtype=npu.DEFAULT_INT,
                                              sep=" ")
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

        :param name: the name of the instance to be loaded
        :param stream: the text stream
        :return: the instance
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

        :param name: the instance name
        :return: the instance

        >>> jssp = Instance.from_resource("demo")
        >>> print(jssp.jobs)
        4
        >>> print(jssp.machines)
        5
        >>> print(jssp)
        demo
        """
        with resources.open_text(package=str(__package__),
                                 resource="demo.txt" if (name == "demo")
                                 else "instances.txt") as stream:
            return Instance.from_stream(name=name, stream=stream)

    @staticmethod
    def list_resources() -> Tuple[str, ...]:
        """
        Get a tuple with all JSSP instances provided in the moptipy resources.

        :return: a tuple with all instance names that are valid parameters
            to :meth:`Instance.from_resource`

        >>> print(Instance.list_resources()[0:3])
        ('abz5', 'abz6', 'abz7')
        """
        return _INSTANCES


def check_instance(inst: Instance) -> Instance:
    """
    Check whether the contents of a JSSP instance are OK.

    This method thoroughly checks the contents of an instance and the
    types of all of its members. If your instances passes this method
    without any error, it is a valid JSSP instance that can be used for
    experiments. All instances in our benchmark set listed above will
    pass this test.

    :param inst: the instance
    :returns: the instance, if its contents are OK
    """
    if not isinstance(inst, Instance):
        raise type_error(inst, "instance", Instance)
    if not isinstance(inst, np.ndarray):
        raise type_error(inst, "instance", np.ndarray)
    if not isinstance(inst.machines, int):
        raise type_error(inst.machines, "inst.machines", int)
    if not isinstance(inst.name, str):
        raise type_error(inst.name, "inst.name", str)
    if (len(inst.name) <= 0) \
            or (inst.name != sanitize_name(inst.name)):
        raise ValueError(f"invalid instance name '{inst.name}'")
    if inst.machines <= 0:
        raise ValueError(f"inst.machines must be > 0, but is {inst.machines}"
                         f" for instance {inst.name}")
    if not isinstance(inst.jobs, int):
        raise type_error(inst.jobs, "inst.jobs", int)
    if inst.machines <= 0:
        raise ValueError(f"inst.jobs must be > 0, but is {inst.jobs}"
                         f" for instance {inst.name}.")
    if not isinstance(inst.makespan_lower_bound, int):
        raise type_error(inst.makespan_lower_bound,
                         f"inst.makespan_lower_bound of {inst.name}", int)
    if inst.makespan_lower_bound <= 0:
        raise ValueError("inst.makespan_lower_bound must be > 0,"
                         f" but is {inst.makespan_lower_bound}"
                         f" for instance {inst.name}.")
    if not isinstance(inst.makespan_upper_bound, int):
        raise type_error(inst.makespan_upper_bound,
                         f"inst.makespan_upper_bound of {inst.name}", int)
    if inst.makespan_upper_bound <= 0:
        raise ValueError("inst.makespan_upper_bound must be > 0,"
                         f" but is {inst.makespan_upper_bound}"
                         f" for instance {inst.name}.")
    if len(inst.shape) != 3:
        raise ValueError(f"inst must be 3d-array, but has shape {inst.shape}"
                         f" for instance {inst.name}.")
    if inst.shape[0] != inst.jobs:
        raise ValueError(
            f"inst.shape[0] must be inst.jobs={inst.jobs}, "
            f"but inst has shape {inst.shape} for instance {inst.name}.")
    if inst.shape[1] != inst.machines:
        raise ValueError(
            f"inst.machines[1] must be inst.machines={inst.machines}, "
            f"but inst has shape {inst.shape} for instance {inst.name}.")
    if inst.shape[2] != 2:
        raise ValueError(
            f"inst.machines[2] must be 2, but inst has shape {inst.shape}"
            f" for instance {inst.name}.")

    for i in range(inst.jobs):
        for j in range(inst.machines):
            machine = inst[i, j, 0]
            if not (0 <= machine < inst.machines):
                raise ValueError(
                    f"encountered machine {machine} for job {i} in "
                    f"operation {j}, but there are only "
                    f"{inst.machines} machines for instance {inst.name}.")
    mslb = compute_makespan_lower_bound(
        machines=inst.machines, jobs=inst.jobs, matrix=inst)
    if mslb > inst.makespan_lower_bound:
        raise ValueError(f"makespan lower bound computed as {mslb},"
                         f"but set to {inst.makespan_upper_bound},"
                         "which is not lower  for instance {inst.name}.")
    msub = sum(inst[:, :, 1].flatten())
    if msub != inst.makespan_upper_bound:
        raise ValueError(f"makespan upper bound computed as {msub}, "
                         f"but set to {inst.makespan_upper_bound}"
                         f" for instance {inst.name}.")
    tub = max([sum(inst[i, :, 1]) for i in range(inst.jobs)])
    if inst.makespan_lower_bound < tub:
        raise ValueError(f"makespan lower bound {inst.makespan_lower_bound} "
                         f"less then longest job duration {tub}"
                         f" for instance {inst.name}.")
    return inst
