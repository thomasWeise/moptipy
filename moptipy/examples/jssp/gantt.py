"""A class for representing Gantt charts as objects."""
import numpy as np

from moptipy.examples.jssp.instance import Instance
from moptipy.utils import nputils


class Gantt:
    """
    A class representing Gantt charts.

    A Gantt chart is a diagram that visualizes when a job on a given machine
    begins or ends. We here represent it as a three-dimensional matrix
    `self.times`. This matrix has one row for each job and one column for each
    machine. In each cell, it holds two values: the start and the end time of
    the job on the machine.
    """

    def __init__(self, instance: Instance) -> None:
        """
        Create a Gantt chart record to hold a solution for a JSSP instance.

        :param Instance instance: the JSSP instance
        """
        if not isinstance(instance, Instance):
            TypeError("Must provide valid JSSP instance, but passed "
                      f"in a {type(instance)}.")
        #: The JSSP Instance to which this Gantt record applies.
        self.instance = instance

        #: A 3d matrix with (start, stop) time for each job on each machine.
        self.times = np.zeros(
            shape=(instance.jobs, instance.machines, 2),
            dtype=nputils.int_range_to_dtype(
                min_value=instance.makespan_lower_bound,
                max_value=instance.makespan_upper_bound))

        #: The makespan of the Gantt chart, an integer.
        self.makespan: int = 0

    def compute_statistics(self) -> None:
        """
        Re-compute all statistics of the Gantt chart.

        This currently only includes the makespan, which simply is the largest
        recorded number in `self.times`.
        """
        self.makespan = int(self.times.max())
