import numpy as np
from moptipy.utils import nputils
from moptipy.examples.jssp.instance import JSSPInstance


class Gantt:
    """
    A Gantt chart is a diagram that visualizes when a job on a given machine
    begins or ends. We here represent it as a three-dimensional matrix
    `self.times`. This matrix has one row for each job and one column for each
    machine. In each cell, it holds two values: the start and the end time of
    the job on the machine.
    """

    def __init__(self, instance: JSSPInstance):
        """
        Create a Gantt chart record to hold a solution for a JSSP instance.
        :param JSSPInstance instance: the JSSP instance
        """
        if not isinstance(instance, JSSPInstance):
            ValueError("Must provide valid JSSP instance, but passed in a '"
                       + str(type(instance)) + "'.")
        self.instance = instance
        """The JSSP Instance to which this Gantt record applies."""
        self.times = np.zeros(
            shape=(instance.jobs, instance.machines, 2),
            dtype=nputils.int_range_to_dtype(
                min_value=instance.makespan_lower_bound,
                max_value=instance.makespan_upper_bound))
        """A 3d matrix holding, for each job on each machine,
        the start and end times."""
        self.makespan = 0
        """The makespan of the chart."""

    def compute_statistics(self):
        """
        Re-compute all statistics of the Gantt chart.
        This currently only includes the makespan, which simply is the largest
        recorded number in `self.times`.
        """
        self.makespan = self.times.max()
