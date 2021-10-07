"""
The Job Shop Scheduling Problem is a good example for optimization tasks.

The JSSP is one of the most well-known combinatorial optimization tasks.
Here we provide a set of benchmark instances and examples for solving this
problem.
"""
from moptipy.examples.jssp.gantt import Gantt
from moptipy.examples.jssp.gantt_space import GanttSpace
from moptipy.examples.jssp.instance import Instance
from moptipy.examples.jssp.makespan import Makespan
from moptipy.examples.jssp.ob_encoding import OperationBasedEncoding

__all__ = (
    "Gantt",
    "GanttSpace",
    "Instance",
    "Makespan",
    "OperationBasedEncoding")
