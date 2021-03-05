"""
In this package, we provide implementations of metaheuristic algorithms.

The black-box methods are given directly in this package, more specialized
algorithms will be placed in sub-packages corresponding to their requirements
for the search- and solution space.
"""

from typing import Final

import moptipy.version
from moptipy.algorithms.hill_climber import HillClimber
from moptipy.algorithms.random_sampling import RandomSampling
from moptipy.algorithms.random_walk import RandomWalk

__version__: Final[str] = moptipy.version.__version__

__all__ = (
    "HillClimber",
    "RandomSampling",
    "RandomWalk")
