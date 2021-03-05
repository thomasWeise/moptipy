"""
moptipy is the *Metaheuristic Optimization in Python* Package.

This is the basic package of our moptipy project.
It provides several sub-packages:

* :py:mod:`moptipy.api` the API for implementing optimization
  algorithms and -problems
* :py:mod:`moptipy.spaces` several pre-defined search spaces
* :py:mod:`moptipy.utils` utility classes for, e.g., logging
  and naming
"""
from typing import Final

import moptipy.version

__version__: Final[str] = moptipy.version.__version__
