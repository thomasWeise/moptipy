"""A class for representing Gantt charts as objects."""
import numpy as np

from moptipy.examples.jssp.instance import Instance


# start book
class Gantt(np.ndarray):
    """
    A class representing Gantt charts.

    A Gantt chart is a diagram that visualizes when a job on a given
    machine begins or ends. We here represent it as a three-dimensional
    matrix. This matrix has one row for each machine and one column for each
    job. In each cell, it holds three values: the job ID, the start, and the
    end time of the job on the machine.
    The Gantt chart has the additional attribute `instance` which references
    the JSSP instance for which the chart is constructed.
    Gantt charts must only be created by an instance of
    :class:`moptipy.examples.jssp.gant_space.GanttSpace`.
    """

    #: the JSSP instance for which the Gantt chart is created
    instance: Instance
# end book

    def __new__(cls, space):
        """
        Create the Gantt chart.

        :param moptipy.examples.jssp.gant_space.GanttSpace space: the Gantt
            space for which the instance is created.
        """
        obj = np.ndarray.__new__(Gantt, space.shape, space.instance.dtype)
        #: store the instance in this object
        obj.instance = space.instance
        return obj
