"""
In this package, some pre-defined search spaces are provided.
"""
from typing import Final, Tuple

import moptipy.version
from moptipy.spaces.bitstrings import BitStrings
from moptipy.spaces.intspace import IntSpace
from moptipy.spaces.permutations import Permutations
from moptipy.spaces.permutationswr import PermutationsWithRepetitions
from moptipy.spaces.vectorspace import VectorSpace

__version__: Final[str] = moptipy.version.__version__

__all__: Final[Tuple[str, ...]] = (
    "BitStrings",
    "IntSpace",
    "Permutations",
    "PermutationsWithRepetitions",
    "VectorSpace")
