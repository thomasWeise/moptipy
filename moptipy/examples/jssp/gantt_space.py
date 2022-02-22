"""Here we implement a space implementation for :class:`Gantt` charts."""
from math import factorial
from typing import Final, Tuple, Set

import numpy as np

from moptipy.api.space import Space
from moptipy.examples.jssp.gantt import Gantt
from moptipy.examples.jssp.instance import Instance, SCOPE_INSTANCE
from moptipy.utils.logger import KeyValueSection
from moptipy.utils.nputils import int_range_to_dtype, KEY_NUMPY_TYPE

#: the array shape
KEY_SHAPE: Final[str] = "shape"


def gantt_space_size(jobs: int, machines: int) -> int:
    """
    Compute the size of the Gantt space.

    :param int jobs: the number of jobs
    :param int machines: the number of machines
    :return: the size of the search
    :rtype: int

    >>> print(gantt_space_size(8, 5))
    106562062388507443200000
    """
    if not isinstance(jobs, int):
        raise TypeError(f"Number of jobs must be int, but is {type(jobs)}.")
    if jobs <= 0:
        raise ValueError(f"Number of jobs must be > 0, but is {jobs}.")
    if not isinstance(machines, int):
        raise TypeError(
            f"Number of machines must be int, but is {type(machines)}.")
    if machines <= 0:
        raise ValueError(
            f"Number of machines must be > 0, but is {machines}.")
    return factorial(jobs) ** machines


# start book
class GanttSpace(Space):
    """A space implementation of for `Gantt` charts."""

    def __init__(self, instance: Instance) -> None:
        # end book
        """
        Create a Gantt chart space.

        :param moptipy.examples.jssp.Instance instance: the JSSP instance
        """
        if not isinstance(instance, Instance):
            ValueError("Must provide valid JSSP instance, "
                       f"but passed in a {type(instance)}.")
        #: The JSSP Instance to which the Gantt record apply.
        self.instance: Final[Instance] = instance  # +book
        #: The shape for the Gantt chart arrays.
        self.shape: Final[Tuple[int, int, int]] = \
            (instance.machines, instance.jobs, 3)
        #: the data to be used for Gantt charts
        self.dtype: Final[np.dtype] = int_range_to_dtype(
            min_value=0, max_value=instance.makespan_upper_bound)
        # function wrapper
        self.copy = np.copyto  # type: ignore

    def create(self) -> Gantt:  # +book
        """
        Create a Gantt chart object without assigning jobs to machines.

        :return: the Gantt chart
        :rtype: np.ndarray
        """
        return Gantt(self)  # +book

    def to_str(self, x: Gantt) -> str:  # +book
        """
        Convert a Gantt chart to a string.

        :param np.ndarray x: the Gantt chart
        :return: a string corresponding to the flattened
            :py:attr:`~Gantt.times` array
        :rtype: str
        """
        return ",".join([str(xx) for xx in np.nditer(x)])  # +book

    def is_equal(self, x1: Gantt, x2: Gantt) -> bool:  # +book
        """
        Check if two Gantt charts have the same contents.

        :param np.ndarray x1: the first chart
        :param np.ndarray x2: the second chart
        :return: `True` if both charts are for the same instance and have the
            same structure
        :rtype: bool
        """
        # start book
        return (x1.instance is x2.instance) and np.array_equal(x1, x2)
        # end book

    def from_str(self, text: str) -> Gantt:  # +book
        """
        Convert a string to a Gantt chart.

        :param str text: the string
        :return: the Gantt chart
        :rtype: np.ndarray
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be str, but is {type(text)}.")
        # start book
        x: Final[Gantt] = self.create()
        np.copyto(x, np.fromstring(text, dtype=self.dtype, sep=",")
                  .reshape(self.shape))
        self.validate(x)  # -book
        return x
        # end book

    def validate(self, x: Gantt) -> None:  # +book
        """
        Check if a Gantt chart if valid and feasible.

        :param np.ndarray x: the Gantt chart
        :raises TypeError: if any component of the chart is of the wrong type
        :raises ValueError: if the Gantt chart is not feasible or the makespan
            is wrong
        """
        # start book
        # Checks if a Gantt chart if valid and feasible.
        if not isinstance(x, Gantt):
            raise TypeError(f"x must be Gantt, but is {type(x)}.")
        # the rest of the checks is not printed for brevity reasons...
        # end book
        inst: Final[Instance] = self.instance
        if inst is not x.instance:
            raise ValueError(
                f"x.instance must be {inst} but is {x.instance}.")
        jobs: Final[int] = inst.jobs
        machines: Final[int] = inst.machines
        if len(x.shape) != 3:
            raise ValueError("x must be three-dimensional, "
                             f"but is {len(x.shape)}-dimensional.")
        if x.shape[0] != machines:
            raise ValueError(
                f"x must have {machines} rows for instance {inst.name}, "
                f"but has {x.shape[0]}.")

        if x.shape[1] != jobs:
            raise ValueError(
                f"x must have {jobs} columns for instance {inst.name}, "
                f"but has {x.shape[1]}.")
        if x.shape[2] != 3:
            raise ValueError("x must have three values per cell, "
                             f"but has {x.shape[2]}.")
        if x.dtype != self.dtype:
            raise ValueError(
                f"x.dtype should be {self.dtype} for instance "
                f"{inst.name} but is {x.dtype}.")

        # now check the sequence on operations per machine
        # we check for overlaps and incorrect operation times
        jobs_done: Final[Set[int]] = set()
        for machinei in range(machines):
            jobs_done.clear()
            last_end = 0
            last_name = "[start]"
            for jobi in range(jobs):
                job = int(x[machinei, jobi, 0])
                start = int(x[machinei, jobi, 1])
                end = int(x[machinei, jobi, 2])
                if not (0 <= job < jobs):
                    raise ValueError(
                        f"job {job} invalid for machine {machinei} on "
                        f"instance {inst.name} with {jobs} jobs: "
                        f"{x[machinei]}")
                if job in jobs_done:
                    raise ValueError(
                        f"job {job} appears twice on machine {machinei} "
                        f"for instance {inst.name}: {x[machinei]}")
                jobs_done.add(job)
                if start < last_end:
                    raise ValueError(
                        f"job {job} starts at {start} on machine {machinei}, "
                        f"for instance {inst.name} but cannot before "
                        f"{last_name}: {x[machinei]}.")
                time = -1
                for z in range(machines):
                    if inst[job, z, 0] == machinei:
                        time = int(inst[job, z, 1])
                        break
                if time < 0:
                    raise ValueError(
                        f"Did not find machine {machinei} for job {job} in "
                        f"instance {inst.name}: {x[machinei]}, "
                        f"{inst[machinei]}.")
                if (end - start) != time:
                    raise ValueError(
                        f"job {job} should need {time} time units on machine "
                        f"{machinei} for instance {inst.name}, but starts at "
                        f"{start} and ends at {end}: {x[machinei]}, "
                        f"{inst[job]}.")

        # now check the single jobs
        # we again check for operation overlaps and incorrect timing
        for jobi in range(jobs):
            done = [(start, end, machine)
                    for machine in range(machines)
                    for (idx, start, end) in x[machine, :, :] if idx == jobi]
            done.sort()

            last_end = 0
            last_machine = "[start]"
            for machinei in range(machines):
                machine, time = inst[jobi, machinei]
                start = int(done[machinei][0])
                end = int(done[machinei][1])
                used_machine = int(done[machinei][2])
                if machine != used_machine:
                    raise ValueError(
                        f"Machine at index {machinei} of job {jobi} "
                        f"must be {machine} for instance {inst.name}, "
                        f"but is {used_machine}.")
                needed = end - start
                if needed != time:
                    raise ValueError(
                        f"Job {jobi} must be processed on {machine} for "
                        f"{time} time units on instance {inst.name}, but "
                        f"only {needed} are used.")
                if needed < 0:
                    raise ValueError(
                        f"Processing time of job {jobi} on machine {machine} "
                        f"cannot be < 0 for instance {inst.name}, but "
                        f"is {needed}.")

                if start < last_end:
                    raise ValueError(
                        f"Processing time window [{start},{end}] on "
                        f"machine {machine} for instance {inst.name}"
                        f"intersects with last operation end {last_end} on "
                        f"machine {last_machine}.")

                last_end = end
                last_machine = machine

        maxtime: Final[int] = int(x[:, -1, 2].max())
        if maxtime < inst.makespan_lower_bound:
            raise ValueError(
                f"Makespan {maxtime} computed, which is smaller than the "
                f"lower bound {inst.makespan_lower_bound} of the JSSP "
                f"instance '{inst}'.")
        if maxtime > inst.makespan_upper_bound:
            raise ValueError(
                f"Makespan {maxtime} computed, which is larger than the "
                f"upper bound {inst.makespan_upper_bound} of the JSSP "
                f"instance '{inst}'.")

    def n_points(self) -> int:
        """
        Get the number of possible Gantt charts without useless delays.

        :return: `factorial(jobs) ** machines`
        :rtype: int

        >>> print(GanttSpace(Instance.from_resource("demo")).n_points())
        7962624
        """
        return gantt_space_size(self.instance.jobs, self.instance.machines)

    def __str__(self) -> str:
        """
        Get the name of the Gantt space.

        :return: the name
        :rtype: str

        >>> space = GanttSpace(Instance.from_resource("abz7"))
        >>> print(space)
        gantt_abz7
        """
        return f"gantt_{self.instance}"

    def log_parameters_to(self, logger: KeyValueSection) -> None:
        """
        Log the parameters of the Gantt space to the given logger.

        :param moptipy.utils.KeyValueSection logger: the logger
        """
        super().log_parameters_to(logger)
        logger.key_value(KEY_SHAPE, repr(self.shape))
        logger.key_value(KEY_NUMPY_TYPE, self.dtype.char)
        with logger.scope(SCOPE_INSTANCE) as kv:
            self.instance.log_parameters_to(kv)
