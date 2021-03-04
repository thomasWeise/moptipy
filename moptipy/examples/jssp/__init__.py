"""
The Job Shop Scheduling Problem is one of the most well-known combinatorial
optimization tasks.
Here we provide a set of benchmark instances and examples for solving this
problem.
"""
from typing import Final, Tuple

import moptipy.version
from moptipy.examples.jssp.gantt import Gantt
from moptipy.examples.jssp.gantt_space import GanttSpace
from moptipy.examples.jssp.instance import JSSPInstance
from moptipy.examples.jssp.makespan import Makespan
from moptipy.examples.jssp.ob_encoding import OperationBasedEncoding

__version__: Final[str] = moptipy.version.__version__

__all__: Final[Tuple[str, ...]] = (
    "Gantt",
    "GanttSpace",
    "JSSPInstance",
    "Makespan",
    "OperationBasedEncoding")
