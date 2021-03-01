from moptipy.api.space import Space
from moptipy.examples.jssp.gantt import Gantt
from moptipy.examples.jssp.instance import JSSPInstance
from moptipy.utils.logger import KeyValueSection
import numpy as np
from math import factorial


class GanttSpace(Space):
    """
    An implementation of :py:class:`~moptipy.api.space.Space` for
    :py:class:`~moptipy.examples.jssp.gantt.Gantt` charts.
    """

    def __init__(self, instance: JSSPInstance):
        """
        Create a Gantt chart space
        :param JSSPInstance instance: the JSSP instance
        """
        if not isinstance(instance, JSSPInstance):
            ValueError("Must provide valid JSSP instance, but passed in a '"
                       + str(type(instance)) + "'.")
        self.instance = instance
        """The JSSP Instance to which the Gantt record apply."""

    def create(self):
        return Gantt(self.instance)

    def copy(self, source: Gantt, dest: Gantt):
        if dest.instance != source.instance:
            raise ValueError("Instances of source and dest must be the same.")
        np.copyto(dest.times, source.times)
        dest.makespan = source.instance

    def to_str(self, x: Gantt) -> str:
        return ",".join([str(xx) for xx in x.times.flatten()])

    def is_equal(self, x1: Gantt, x2: Gantt) -> bool:
        return (x1.instance == x2.instance) and \
            np.array_equal(x1.times, x2.times)

    def from_str(self, text: str) -> Gantt:
        x = self.create()
        np.copyto(x.times,
                  np.fromstring(text, dtype=x.times.dtype, sep=",")
                  .reshape(x.times.shape))
        x.compute_statistics()
        return x

    def validate(self, x: Gantt):
        if not isinstance(x.instance, JSSPInstance):
            raise ValueError("Invalid instance, not a JSSP instance, but a '"
                             + str(type(x.instance)) + "'.")

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
        return factorial(self.instance.jobs) ** self.instance.machines

    def get_name(self):
        return "gantt_" + self.instance.get_name()

    def log_parameters_to(self, logger: KeyValueSection):
        super().log_parameters_to(logger)
        with logger.scope(JSSPInstance.SCOPE_INSTANCE) as kv:
            self.instance.log_parameters_to(kv)
