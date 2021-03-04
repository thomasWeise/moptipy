"""
The API for implementing optimization algorithms and -problems.

This package provides two things:

* the basic, abstract API for implementing optimization algorithms and
  problems
* the abstraction and implementation of black-box processes in form of
  :py:class:`~moptipy.api.Process` and its implementations

The former helps us to implement and plug together different components
of optimization problems and optimization algorithms.
The latter allows us to apply algorithms to problems and to collect the
results in a transparent way.
It also permits logging of the algorithm progress or even the change of
dynamic parameters.
"""

from typing import Final, Tuple

import moptipy.version
from moptipy.api.algorithm import Algorithm, Algorithm0, Algorithm1, \
    Algorithm2, CallableAlgorithm
from moptipy.api.component import Component
from moptipy.api.encoding import Encoding
from moptipy.api.experiment import Experiment, run_experiment
from moptipy.api.objective import CallableObjective, Objective
from moptipy.api.operators import Op0, Op1, Op2
from moptipy.api.process import Process
from moptipy.api.space import Space

__version__: Final[str] = moptipy.version.__version__

__all__: Final[Tuple[str, ...]] = (
    "Algorithm",
    "Algorithm0",
    "Algorithm1",
    "Algorithm2",
    "CallableAlgorithm",
    "CallableObjective",
    "Component",
    "Encoding",
    "Experiment",
    "Objective",
    "Op0",
    "Op1",
    "Op2",
    "Process",
    "run_experiment",
    "Space")
