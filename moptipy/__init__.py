"""
moptipy is the *Metaheuristic Optimization in Python* Package.

This is the basic package of our moptipy project.
It provides several sub-packages:

* :mod:`moptipy.algorithms` hosts implementations of several metaheuristic
  optimization algorithms
* :mod:`moptipy.api` defines the API for implementing optimization algorithms
  and -problems
* :mod:`moptipy.evaluation` offers evaluation utilities for the results of
  experiments
* :mod:`moptipy.mo` provides utilities for multi-objective optimization
* :mod:`moptipy.operators` implements search operators for various search
  spaces
* :mod:`moptipy.spaces` includes several pre-defined search spaces
* :mod:`moptipy.tests` provides several unit tests for moptipy components
* :mod:`moptipy.utils` has utility classes for, e.g., logging and naming
"""
from typing import Final

import moptipy.version

__version__: Final[str] = moptipy.version.__version__
