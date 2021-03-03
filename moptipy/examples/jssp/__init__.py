"""
The Job Shop Scheduling Problem is one of the most well-known combinatorial
optimization tasks.
Here we provide a set of benchmark instances and examples for solving this
problem.
"""
# noinspection PyUnresolvedReferences
from moptipy.version import __version__
from moptipy.examples.jssp.instance import JSSPInstance
from moptipy.examples.jssp.gantt import Gantt
from moptipy.examples.jssp.gantt_space import GanttSpace
from moptipy.examples.jssp.ob_encoding import OperationBasedEncoding
from moptipy.examples.jssp.makespan import Makespan

__all__ = ("Gantt",
           "GanttSpace",
           "JSSPInstance",
           "Makespan",
           "OperationBasedEncoding")
