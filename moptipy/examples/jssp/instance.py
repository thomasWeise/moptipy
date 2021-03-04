"""
Here we provide a representation for JSSP instances.
"""
from importlib import resources
from typing import Final, List, Tuple, Optional

import numpy as np

from moptipy.api import Component
from moptipy.utils import logging
from moptipy.utils import nputils
from moptipy.utils.logger import KeyValueSection
from moptipy.utils.nputils import int_range_to_dtype


class JSSPInstance(Component):
    """An instance of the Job Shop Scheduling Problem."""

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

    def __init__(self,
                 name: str,
                 machines: int,
                 jobs: int,
                 matrix: np.ndarray,
                 makespan_lower_bound: int = None) -> None:
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

        if (not isinstance(machines, int)) or (machines < 1):
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

        # We now compute the lower bound for the makespan based on the
        # algorithm by Taillard
        usedmachines = np.zeros(machines, np.dtype(np.bool_))
        jobtimes = np.zeros(jobs, nputils.DEFAULT_INT)
        machinetimes = np.zeros(machines, nputils.DEFAULT_INT)
        machine1 = nputils.intmax(machines, nputils.DEFAULT_INT)
        machine2 = nputils.intmax(machines, nputils.DEFAULT_INT)

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
            jobremaining = jobtime
            j = len(row)
            for i in range(machines):
                j -= 2
                machine = row[j]
                time = row[j + 1]
                machine2[machine] = min(machine2[machine],
                                        jobtime - jobremaining)
                jobremaining -= time

            if not all(usedmachines):
                ValueError("Some machines not used in a job in instance '"
                           + name + "'.")
            jobidx += 1

        ms_lower_bound = max(int(jobtimes.max()),
                             int((machine1 + machine2 + machinetimes).max()))
        if ms_lower_bound <= 0:
            ValueError("Computed makespan lower bound must not be <= 0, "
                       "but is '" + str(ms_lower_bound) + "'.")
        ms_upper_bound = int(jobtimes.sum())
        if ms_upper_bound < ms_lower_bound:
            ValueError("Computed makespan upper bound " + str(ms_upper_bound)
                       + "must be <= than computed lower bound"
                       + str(ms_lower_bound) + ".")

        if makespan_lower_bound is None:
            makespan_lower_bound = ms_lower_bound
        else:
            if (not isinstance(makespan_lower_bound, int)) or \
                    (makespan_lower_bound <= 0):
                ValueError("If specified, makespan_lower_bound must be "
                           "positive integer, but is "
                           + str(makespan_lower_bound) + " in instance '"
                           + name + "'.")
            if makespan_lower_bound < ms_lower_bound:
                ValueError("If specified, makespan_lower_bound must be >= "
                           + str(ms_lower_bound) + ", but is "
                           + str(makespan_lower_bound) + " in instance '"
                           + name + "'.")
            if makespan_lower_bound > ms_upper_bound:
                ValueError("If specified, makespan_lower_bound must be <= "
                           + str(ms_upper_bound) + ", but is "
                           + str(makespan_lower_bound) + " in instance '"
                           + name + "'.")

        self.makespan_lower_bound = makespan_lower_bound
        """The lower bound of the makespan for the JSSP instance."""

        self.makespan_upper_bound = ms_upper_bound
        """The upper bound of the makespan for the JSSP instance."""

    def get_name(self) -> str:
        return self.name

    def log_parameters_to(self, logger: KeyValueSection) -> None:
        super().log_parameters_to(logger)
        logger.key_value(JSSPInstance.MACHINES, self.machines)
        logger.key_value(JSSPInstance.JOBS, self.jobs)
        logger.key_value(JSSPInstance.MAKESPAN_LOWER_BOUND,
                         self.makespan_lower_bound)
        logger.key_value(JSSPInstance.MAKESPAN_UPPER_BOUND,
                         self.makespan_upper_bound)

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
        jobs_machines_txt = rows[1]

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
    def from_resource(name: str) -> 'JSSPInstance':
        """
        Load one of the JSSP instances provided as part of this example
        package.
        :param str name: the instance name
        :return: the instance
        :rtype: JSSPInstance
        """
        with resources.open_text(package=str(__package__),
                                 resource="demo.txt" if (name == "demo")
                                 else "instances.txt") as stream:
            return JSSPInstance.from_stream(name=name, stream=stream)

    @staticmethod
    def list_resources() -> Tuple[str, ...]:
        """
        Obtain a tuple with all the instances provided in the resources
        shipping with moptipy
        :return: a tuple with all instance names that are valid parameters
        to :meth:`JSSPInstance.from_resource`
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
