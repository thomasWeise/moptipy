"""In this package, some pre-defined search spaces are provided."""
from moptipy.spaces.bitstrings import BitStrings
from moptipy.spaces.intspace import IntSpace
from moptipy.spaces.permutations import Permutations
from moptipy.spaces.permutationswr import PermutationsWithRepetitions
from moptipy.spaces.vectorspace import VectorSpace

__all__ = (
    "BitStrings",
    "IntSpace",
    "Permutations",
    "PermutationsWithRepetitions",
    "VectorSpace")
