"""
Here an objective function for minimizing the makespan of Gantt charts is
implemented.
"""
from moptipy.api.objective import Objective
from moptipy.examples.jssp.gantt import Gantt
from moptipy.examples.jssp.instance import JSSPInstance


class Makespan(Objective):
    """This objective function returns the makespan of a Gantt chart."""

    def __init__(self, instance: JSSPInstance) -> None:
        super().__init__()
        if not isinstance(instance, JSSPInstance):
            raise ValueError("Must provide JSSPInstance, but got '"
                             + str(type(instance)) + "'.")
        self.__instance = instance

    def evaluate(self, x: Gantt) -> int:
        return int(x.makespan)

    def lower_bound(self) -> int:
        return self.__instance.makespan_lower_bound

    def upper_bound(self) -> int:
        return self.__instance.makespan_upper_bound

    def get_name(self) -> str:
        return "makespan"
