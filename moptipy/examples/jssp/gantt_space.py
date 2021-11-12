"""Here we implement a space implementation for :class:`Gantt` charts."""
from math import factorial
from typing import Final

import numpy as np

from moptipy.api.space import Space
from moptipy.examples.jssp.gantt import Gantt
from moptipy.examples.jssp.instance import Instance, SCOPE_INSTANCE
from moptipy.utils.logger import KeyValueSection


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

    def create(self) -> Gantt:  # +book
        """
        Create a Gantt chart object without assigning jobs to machines.

        :return: the Gantt chart
        :rtype: moptipy.examples.jssp.Gantt
        """
        return Gantt(self.instance)  # +book

    def copy(self, source: Gantt, dest: Gantt) -> None:  # +book
        """
        Copy the contents of one Gantt chart to another.

        :param source: the source chart
        :param dest: the destination chart
        """
        if dest.instance != source.instance:
            raise ValueError("Instances of source and dest must be the same.")
        np.copyto(dest.times, source.times)  # +book

    def to_str(self, x: Gantt) -> str:  # +book
        """
        Convert a Gantt chart to a string.

        :param moptipy.examples.jssp.Gantt x: the Gantt chart
        :return: a string corresponding to the flattened
            :py:attr:`~Gantt.times` array
        :rtype: str
        """
        return ",".join([str(xx) for xx in x.times.flatten()])  # +book

    def is_equal(self, x1: Gantt, x2: Gantt) -> bool:  # +book
        """
        Check if two Gantt charts have the same contents.

        :param moptipy.examples.jssp.Gantt x1: the first chart
        :param moptipy.examples.jssp.Gantt x2: the second chart
        :return: `True` if both charts are for the same instance and have the
            same structure
        :rtype: bool
        """
        # start book
        return (x1.instance == x2.instance) and \
            np.array_equal(x1.times, x2.times)
        # end book

    def from_str(self, text: str) -> Gantt:  # +book
        """
        Convert a string to a Gantt chart.

        :param str text: the string
        :return: the Gantt chart
        :rtype: moptipy.examples.jssp.Gantt
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be str, but is {type(text)}.")
        # start book
        x: Final[Gantt] = self.create()
        np.copyto(x.times,
                  np.fromstring(text, dtype=x.times.dtype, sep=",")
                  .reshape(x.times.shape))
        self.validate(x)  # -book
        return x
        # end book

    def validate(self, x: Gantt) -> None:  # +book
        """
        Check if a Gantt chart if valid and feasible.

        :param moptipy.examples.jssp.Gantt x: the Gantt chart
        :raises TypeError: if any component of the chart is of the wrong type
        :raises ValueError: if the Gantt chart is not feasible or the makespan
            is wrong
        """
        # checks if a Gantt chart if valid and feasible.  # +book
        if not isinstance(x.instance, Instance):
            raise TypeError("Invalid instance, not a JSSP instance, "
                            f"but a {type(x.instance)}.")
        if not isinstance(x.times, np.ndarray):
            raise TypeError(
                f"x.times must be numpy.ndarray, but is {type(x.times)}.")
        if not isinstance(x.instance, Instance):
            raise TypeError("x.instance must be a Instance, "
                            f"but is {type(x.instance)}.")
        if not isinstance(x.instance.matrix, np.ndarray):
            raise TypeError("x.instance.matrix must be numpy.ndarray, "
                            f"but is {type(x.instance.matrix)}.")
        if not isinstance(x.instance.jobs, int):
            raise TypeError("x.instance.jobs must be int, "
                            f"but is {type(x.instance.jobs)}.")
        if not isinstance(x.instance.machines, int):
            raise TypeError("x.instance.machines must be int, "
                            f"but is {type(x.instance.machines)}.")

        if x.times.shape[0] != x.instance.jobs:
            raise ValueError(f"times matrix must have {x.instance.jobs} "
                             f"rows, but has {x.times.shape[0]}.")

        if x.times.shape[1] != x.instance.machines:
            raise ValueError(f"times matrix must have {x.instance.machines} "
                             f"rows, but has {x.times.shape[1]}.")

        if x.times.shape[2] != 2:
            raise ValueError("times matrix must have two values per cell, "
                             f"but has {x.times.shape[2]}.")

        for jobi in range(x.instance.jobs):
            row = x.times[jobi]
            srow = [(row[i, 0], row[i, 1], i) for i in range(len(row))]
            srow.sort()
            data = x.instance.matrix[jobi]
            last_end = 0
            last_machine = "[start]"
            for machinei in range(x.instance.machines):
                machine = data[machinei * 2]
                time = data[1 + (machinei * 2)]
                if machine != (srow[machinei])[2]:
                    raise ValueError(
                        f"Machine at index {machinei} of job {jobi} "
                        f"must be {machine}, but is {(srow[machinei])[2]}.")
                start = (srow[machinei])[0]
                end = (srow[machinei])[1]
                needed = end - start
                if needed != time:
                    raise ValueError(
                        f"Job {jobi} must be processed on {machine} for "
                        f"{time} time units, but only {needed} are used.")
                if needed < 0:
                    raise ValueError(
                        f"Processing time of job {jobi} on machine {machine} "
                        f"cannot be < 0, but is {needed}.")

                if start < last_end:
                    raise ValueError(
                        f"Processing time window [{start},{end}] on "
                        f"machine {machine} intersects with last operation "
                        f"end {last_end} on machine {last_machine}.")

                last_end = end
                last_machine = machine

        maxtime: Final[int] = int(x.times.max())
        if maxtime < x.instance.makespan_lower_bound:
            raise ValueError(
                f"Makespan {maxtime} computed, which is smaller than the "
                f"lower bound {x.instance.makespan_lower_bound} of the JSSP "
                f"instance '{x.instance.get_name()}'.")
        if maxtime > x.instance.makespan_upper_bound:
            raise ValueError(
                f"Makespan {maxtime} computed, which is larger than the "
                f"upper bound {x.instance.makespan_upper_bound} of the JSSP "
                f"instance '{x.instance.get_name()}'.")

    def scale(self) -> int:
        """
        Get the number of possible Gantt charts without useless delays.

        :return: `factorial(jobs) ** machines`
        :rtype: int

        >>> print(GanttSpace(Instance.from_resource("demo")).scale())
        7962624
        """
        return factorial(self.instance.jobs) ** self.instance.machines

    def get_name(self) -> str:
        """
        The name of the Gantt space.

        :return: the name
        :rtype: str

        >>> space = GanttSpace(Instance.from_resource("abz7"))
        >>> print(space.get_name())
        gantt_abz7
        """
        return f"gantt_{self.instance.get_name()}"

    def log_parameters_to(self, logger: KeyValueSection) -> None:
        """
        Log the parameters of the Gantt space to the given logger.

        :param moptipy.utils.KeyValueSection logger: the logger
        """
        super().log_parameters_to(logger)
        with logger.scope(SCOPE_INSTANCE) as kv:
            self.instance.log_parameters_to(kv)
