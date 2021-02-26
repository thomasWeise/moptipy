"""
In this package, some pre-defined search spaces are provided.
"""
# noinspection PyUnresolvedReferences
from moptipy.version import __version__
from moptipy.spaces.vectorspace import VectorSpace
from moptipy.spaces.intspace import IntSpace
from moptipy.spaces.bitstrings import BitStrings
from moptipy.spaces.permutations import Permutations
from moptipy.spaces.permutationswr import PermutationsWithRepetitions

__all__ = ["BitStrings",
           "IntSpace",
           "Permutations",
           "PermutationsWithRepetitions",
           "VectorSpace"]
