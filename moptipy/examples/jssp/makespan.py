"""An objective function for minimizing the makespan of Gantt charts."""
from typing import Final

from moptipy.api.objective import Objective
from moptipy.examples.jssp.gantt import Gantt
from moptipy.examples.jssp.instance import Instance


# start book
class Makespan(Objective):
    """This objective function returns the makespan of a Gantt chart."""

# end book
    def __init__(self, instance: Instance) -> None:
        """
        Initialize the makespan objective function.

        :param moptipy.examples.jssp.Instance instance: the instance to
            load the bounds from
        """
        super().__init__()
        if not isinstance(instance, Instance):
            raise TypeError(
                f"Must provide Instance, but got '{type(instance)}'.")
        self.__instance: Final[Instance] = instance

    # start book
    def evaluate(self, x: Gantt) -> int:
        """
        Get the makespan corresponding to a given :class:`Gantt` chart.

        :param moptipy.examples.jssp.Gantt x: the Gantt chart.
        :return: the maximum of any time stored in the chart
        """
        return int(x.times.max())
        # end book

    def lower_bound(self) -> int:
        """
        Get the lower bound of the makespan.

        :return: the lower bound
        :rtype: int
        """
        return self.__instance.makespan_lower_bound

    def upper_bound(self) -> int:
        """
        Get the upper bound of the makespan.

        :return: the sum of all job execution times.
        :rtype: int
        """
        return self.__instance.makespan_upper_bound

    def get_name(self) -> str:
        """
        Get the name of the makespan objective function.

        :return: `makespan`
        :rtype: str
        """
        return "makespan"
