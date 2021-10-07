"""
In this package, we provide implementations of metaheuristic algorithms.

The black-box methods are given directly in this package, more specialized
algorithms will be placed in sub-packages corresponding to their requirements
for the search- and solution space.
"""

from moptipy.algorithms.ea1p1 import EA1p1
from moptipy.algorithms.hill_climber import HillClimber
from moptipy.algorithms.random_sampling import RandomSampling
from moptipy.algorithms.random_walk import RandomWalk
from moptipy.algorithms.single_random_sample import SingleRandomSample

__all__ = (
    "EA1p1",
    "HillClimber",
    "RandomSampling",
    "RandomWalk",
    "SingleRandomSample")
