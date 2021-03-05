"""Here we implement a space implementation for :class:`Gantt` charts."""
from math import factorial

import numpy as np

from moptipy.api.space import Space
from moptipy.examples.jssp.gantt import Gantt
from moptipy.examples.jssp.instance import JSSPInstance
from moptipy.utils.logger import KeyValueSection


class GanttSpace(Space):
    """A space implementation of for `Gantt` charts."""

    def __init__(self, instance: JSSPInstance) -> None:
        """
        Create a Gantt chart space.

        :param moptipy.examples.jssp.JSSPInstance instance: the JSSP instance
        """
        if not isinstance(instance, JSSPInstance):
            ValueError("Must provide valid JSSP instance, but passed in a '"
                       + str(type(instance)) + "'.")
        self.instance = instance
        """The JSSP Instance to which the Gantt record apply."""

    def create(self) -> Gantt:
        """
        Create a Gantt chart object without assigning jobs to machines.

        :return: the Gantt chart
        :rtype: moptipy.examples.jssp.Gantt
        """
        return Gantt(self.instance)

    def copy(self, source: Gantt, dest: Gantt) -> None:
        """
        Copy the contents of one Gantt chart to another.

        :param source: the source chart
        :param dest: the destination chart
        """
        if dest.instance != source.instance:
            raise ValueError("Instances of source and dest must be the same.")
        np.copyto(dest.times, source.times)
        dest.makespan = source.makespan

    def to_str(self, x: Gantt) -> str:
        """
        Convert a Gantt chart to a string.

        :param moptipy.examples.jssp.Gantt x: the Gantt chart
        :return: a string corresponding to the flattened
            :py:attr:`~Gantt.times` array
        :rtype: str
        """
        return ",".join([str(xx) for xx in x.times.flatten()])

    def is_equal(self, x1: Gantt, x2: Gantt) -> bool:
        """
        Check if two Gantt charts have the same contents.

        :param moptipy.examples.jssp.Gantt x1: the first chart
        :param moptipy.examples.jssp.Gantt x2: the second chart
        :return: `True` if both charts are for the same instance and have the
            same structure
        :rtype: bool
        """
        return (x1.instance == x2.instance) and \
            np.array_equal(x1.times, x2.times)

    def from_str(self, text: str) -> Gantt:
        """
        Convert a string to a Gantt chart.

        :param str text: the string
        :return: the Gantt chart
        :rtype: moptipy.examples.jssp.Gantt
        """
        if not (isinstance(text, str)):
            raise TypeError("text must be str, but is "
                            + str(type(text)) + ".")
        x = self.create()
        np.copyto(x.times,
                  np.fromstring(text, dtype=x.times.dtype, sep=",")
                  .reshape(x.times.shape))
        x.compute_statistics()
        self.validate(x)
        return x

    def validate(self, x: Gantt) -> None:
        """
        Validate a Gantt chart `x` and raise errors if it is invalid.

        :param moptipy.examples.jssp.Gantt x: the Gantt chart
        :raises TypeError: if any component of the chart is of the wrong type
        :raises ValueError: if the Gantt chart is not feasible or the makespan
            is wrong
        """
        if not isinstance(x.instance, JSSPInstance):
            raise TypeError("Invalid instance, not a JSSP instance, but a '"
                            + str(type(x.instance)) + "'.")
        if not isinstance(x.times, np.ndarray):
            raise TypeError("x.times must be numpy.ndarray, but is "
                            + str(type(x.times)) + ",")
        if not isinstance(x.makespan, int):
            raise TypeError("x.makespan must be int, but is "
                            + str(type(x.makespan)) + ".")
        if not isinstance(x.instance, JSSPInstance):
            raise TypeError("x.instance must be JSSPInstance, but is "
                            + str(type(x.instance)) + ".")
        if not isinstance(x.instance.matrix, np.ndarray):
            raise TypeError("x.instance.matrix must be numpy.ndarray, but is "
                            + str(type(x.instance.matrix)) + ",")
        if not isinstance(x.instance.jobs, int):
            raise TypeError("x.instance.jobs must be int, but is "
                            + str(type(x.instance.jobs)) + ".")
        if not isinstance(x.instance.machines, int):
            raise TypeError("x.instance.machines must be int, but is "
                            + str(type(x.instance.machines)) + ".")

        if x.times.shape[0] != x.instance.jobs:
            raise ValueError("times matrix must have "
                             + str(x.instance.jobs) + " rows, but has "
                             + str(x.times.shape[0]) + ".")

        if x.times.shape[1] != x.instance.machines:
            raise ValueError("times matrix must have "
                             + str(x.instance.machines)
                             + " rows, but has " + str(x.times.shape[1])
                             + ".")

        if x.times.shape[2] != 2:
            raise ValueError(
                "times matrix must have two values per cell, but has "
                + str(x.times.shape[2]) + ".")

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
                        "Machine at index " + str(machinei)
                        + " of job " + str(jobi) + " must be "
                        + str(machine) + ", but is "
                        + str((srow[machinei])[2]) + ".")
                start = (srow[machinei])[0]
                end = (srow[machinei])[1]
                needed = end - start
                if needed != time:
                    raise ValueError(
                        "Job " + str(jobi) + " must be processed on "
                        + str(machine) + " for " + str(time)
                        + " time units, but only " + str(needed)
                        + " are used.")
                if needed < 0:
                    raise ValueError(
                        "Processing time of job " + str(jobi) + " on machine "
                        + str(machine) + " cannot be < 0, but is "
                        + str(needed) + ".")

                if start < last_end:
                    raise ValueError(
                        "Processing time window [" + str(start) + ","
                        + str(end) + "] on machine " + str(machine)
                        + " intersects with last operation end "
                        + str(last_end) + " on machine " + str(last_machine)
                        + ".")

                last_end = end
                last_machine = machine

        maxtime = int(x.times.max())
        if x.makespan != maxtime:
            raise ValueError("Cached makespan " + str(x.makespan)
                             + " not equal to actual makespan "
                             + str(maxtime) + ".")
        if maxtime < x.instance.makespan_lower_bound:
            raise ValueError(
                "Makespan " + str(maxtime)
                + " computed, which is smaller than the lower bound "
                + str(x.instance.makespan_lower_bound)
                + " of the JSSP instance '"
                + x.instance.get_name() + "'.")
        if maxtime > x.instance.makespan_upper_bound:
            raise ValueError(
                "Makespan " + str(maxtime)
                + " computed, which is larger than the upper bound "
                + str(x.instance.makespan_upper_bound)
                + " of the JSSP instance '"
                + x.instance.get_name() + "'.")

    def scale(self) -> int:
        """
        Get the number of possible Gantt charts without useless delays.

        :return: `factorial(jobs) ** machines`
        :rtype: int

        >>> print(GanttSpace(JSSPInstance.from_resource("demo")).scale())
        7962624
        """
        return factorial(self.instance.jobs) ** self.instance.machines

    def get_name(self) -> str:
        """
        The name of the Gantt space.

        :return: the name
        :rtype: str

        >>> space = GanttSpace(JSSPInstance.from_resource("abz7"))
        >>> print(space.get_name())
        gantt_abz7
        """
        return "gantt_" + self.instance.get_name()

    def log_parameters_to(self, logger: KeyValueSection) -> None:
        """
        Log the parameters of the Gantt space to the given logger.

        :param moptipy.utils.KeyValueSection logger: the logger
        """
        super().log_parameters_to(logger)
        with logger.scope(JSSPInstance.SCOPE_INSTANCE) as kv:
            self.instance.log_parameters_to(kv)
