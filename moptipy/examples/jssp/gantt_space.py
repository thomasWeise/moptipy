"""Here we implement a space implementation for `Gantt` charts."""
from math import factorial
from typing import Final

import numpy as np

from moptipy.api.space import Space
from moptipy.examples.jssp.gantt import Gantt
from moptipy.examples.jssp.instance import SCOPE_INSTANCE, Instance
from moptipy.utils.logger import CSV_SEPARATOR, KeyValueLogSection
from moptipy.utils.nputils import (
    KEY_NUMPY_TYPE,
    int_range_to_dtype,
    val_numpy_type,
)
from moptipy.utils.types import type_error

#: the array shape
KEY_SHAPE: Final[str] = "shape"


def gantt_space_size(jobs: int, machines: int) -> int:
    """
    Compute the size of the Gantt space.

    :param jobs: the number of jobs
    :param machines: the number of machines
    :return: the size of the search

    >>> print(gantt_space_size(8, 5))
    106562062388507443200000
    """
    if not isinstance(jobs, int):
        raise type_error(jobs, "number of jobs", int)
    if jobs <= 0:
        raise ValueError(f"Number of jobs must be > 0, but is {jobs}.")
    if not isinstance(machines, int):
        raise type_error(machines, "number of machines", int)
    if machines <= 0:
        raise ValueError(
            f"Number of machines must be > 0, but is {machines}.")
    return factorial(jobs) ** machines


# start book
class GanttSpace(Space):
    """An implementation of the `Space` API of for `Gantt` charts."""

    def __init__(self, instance: Instance) -> None:
        # end book
        """
        Create a Gantt chart space.

        :param instance: the JSSP instance
        """
        if not isinstance(instance, Instance):
            ValueError("Must provide valid JSSP instance, "
                       f"but passed in a {type(instance)}.")
        #: The JSSP Instance to which the Gantt record apply.
        self.instance: Final[Instance] = instance  # +book
        #: The shape for the Gantt chart arrays.
        self.shape: Final[tuple[int, int, int]] = (  # +book
            instance.machines, instance.jobs, 3)  # +book
        #: the data to be used for Gantt charts
        self.dtype: Final[np.dtype] = int_range_to_dtype(
            min_value=0, max_value=instance.makespan_upper_bound)
        #: fast call function forward
        self.copy = np.copyto  # type: ignore # +book

    def create(self) -> Gantt:  # +book
        """
        Create a Gantt chart object without assigning jobs to machines.

        :return: the Gantt chart
        """
        return Gantt(self)  # +book

    def to_str(self, x: Gantt) -> str:  # +book
        """
        Convert a Gantt chart to a string.

        :param x: the Gantt chart
        :return: a string corresponding to the flattened `Gantt` array
        """
        return CSV_SEPARATOR.join([str(xx) for xx in np.nditer(x)])  # +book

    def is_equal(self, x1: Gantt, x2: Gantt) -> bool:  # +book
        """
        Check if two Gantt charts have the same contents.

        :param x1: the first chart
        :param x2: the second chart
        :return: `True` if both charts are for the same instance and have the
            same structure
        """
        # start book
        return (x1.instance is x2.instance) and np.array_equal(x1, x2)
        # end book

    def from_str(self, text: str) -> Gantt:  # +book
        """
        Convert a string to a Gantt chart.

        :param text: the string
        :return: the Gantt chart
        """
        if not isinstance(text, str):
            raise type_error(text, "instance text", str)
        # start book
        x: Final[Gantt] = self.create()
        np.copyto(x, np.fromstring(
            text, dtype=self.dtype, sep=CSV_SEPARATOR)
            .reshape(self.shape))
        self.validate(x)  # -book
        return x
        # end book

    def validate(self, x: Gantt) -> None:  # +book
        """
        Check if a Gantt chart if valid and feasible.

        This means that the operations of the jobs must appear in the right
        sequences and must not intersect in any way.
        The only exception are operations that need 0 time units. They are
        permitted to appear wherever.

        :param x: the Gantt chart
        :raises TypeError: if any component of the chart is of the wrong type
        :raises ValueError: if the Gantt chart is not feasible or the makespan
            is wrong
        """
        # start book
        # Checks if a Gantt chart if valid and feasible.
        if not isinstance(x, Gantt):
            raise type_error(x, "x", Gantt)
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
        jobs_done: Final[set[int]] = set()
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
                if time <= 0:
                    continue  # job requires zero time, can skip
                if start < last_end:
                    raise ValueError(
                        f"job {job} starts at {start} on machine {machinei}, "
                        f"for instance {inst.name} but cannot before "
                        f"{last_name}: {x[machinei]}.")
                last_end = end

        # now check the single jobs
        # we again check for operation overlaps and incorrect timing
        for jobi in range(jobs):
            done = [(start, end, machine)
                    for machine in range(machines)
                    for (idx, start, end) in x[machine, :, :] if idx == jobi]
            done.sort()
            if len(done) != machines:
                raise ValueError(
                    f"Job {jobi} appears only {len(done)} times instead of "
                    f"{machines} times on instance {inst.name}.")

            # we allow operations of length 0 to appear at any position
            last_end = 0
            last_machine = "[start]"
            done_i = 0
            machine_i = 0
            while True:
                if machine_i < machines:
                    machine, time = inst[jobi, machine_i]
                else:
                    machine = time = -1
                if done_i < machines:
                    start = int(done[machine_i][0])
                    end = int(done[machine_i][1])
                    used_machine = int(done[machine_i][2])
                else:
                    start = end = used_machine = -1

                needed = end - start
                if needed < 0:
                    raise ValueError(
                        f"Operation {machine_i} of job {jobi} scheduled "
                        f"from {start} to {end}?")
                if machine != used_machine:
                    # This is only possible for operations that require zero
                    # time units. We will skip such operations in the checks.
                    if (time == 0) and (machine != -1):
                        machine_i += 1
                        continue
                    if (needed == 0) and (machine != -1):
                        done_i += 1
                        continue
                    raise ValueError(
                        f"Machine at index {done_i} of job {jobi} "
                        f"must be {machine} for instance {inst.name}, "
                        f"for {time} time units, but is {used_machine}"
                        f"from {start} to {end}.")
                if machine == -1:
                    if (machine_i < machines) or (done_i < machines):
                        raise ValueError(  # this should never be possible
                            f"Done {machine_i + 1} machines for job {jobi}, "
                            f"which has {done_i + 1} operations done??")
                    break  # we can stop

                # ok, we are at a regular operation
                if needed != time:
                    raise ValueError(
                        f"Job {jobi} must be processed on {machine} for "
                        f"{time} time units on instance {inst.name}, but "
                        f"only {needed} are used from {start} to {end}.")
                if needed < 0:
                    raise ValueError(
                        f"Processing time of job {jobi} on machine {machine} "
                        f"cannot be < 0 for instance {inst.name}, but "
                        f"is {needed}.")

                if start < last_end:
                    raise ValueError(
                        f"Processing time window [{start},{end}] on "
                        f"machine {machine} for job {jobi} on instance "
                        f"{inst.name} intersects with last operation end"
                        f"{last_end} on machine {last_machine}.")

                last_end = end
                last_machine = machine
                machine_i += 1
                done_i += 1

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

        >>> print(GanttSpace(Instance.from_resource("demo")).n_points())
        7962624
        """
        return gantt_space_size(self.instance.jobs, self.instance.machines)

    def __str__(self) -> str:
        """
        Get the name of the Gantt space.

        :return: the name

        >>> space = GanttSpace(Instance.from_resource("abz7"))
        >>> print(space)
        gantt_abz7
        """
        return f"gantt_{self.instance}"

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of the Gantt space to the given logger.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        logger.key_value(KEY_SHAPE, repr(self.shape))
        logger.key_value(KEY_NUMPY_TYPE, val_numpy_type(self.dtype))
        with logger.scope(SCOPE_INSTANCE) as kv:
            self.instance.log_parameters_to(kv)
