"""A class for representing Gantt charts as objects."""
from typing import Final

import numpy as np

from moptipy.examples.jssp.instance import Instance
from moptipy.utils import nputils


# start book
class Gantt:
    """
    A class representing Gantt charts.

    A Gantt chart is a diagram that visualizes when a job on a given
    machine begins or ends. We here represent it as a three-dimensional
    matrix `self.times`. This matrix has one row for each job and one
    column for each machine. In each cell, it holds two values: the start
    and the end time of the job on the machine.
    """

    def __init__(self, instance: Instance) -> None:
        """
        Create a Gantt chart record to hold a solution for a JSSP instance.

        :param Instance instance: the JSSP instance
        """
        # end book
        if not isinstance(instance, Instance):
            TypeError("Must provide valid JSSP instance, but passed "
                      f"in a {type(instance)}.")
        #: The JSSP Instance to which this Gantt record applies.
        self.instance: Final[Instance] = instance  # +book

        #: A 3D matrix with (start, stop) time for each job on each machine.
        self.times: Final[np.ndarray] = np.zeros(  # +book
            dtype=nputils.int_range_to_dtype(
                min_value=0, max_value=instance.makespan_upper_bound),
            shape=(instance.jobs, instance.machines, 2))  # +book
