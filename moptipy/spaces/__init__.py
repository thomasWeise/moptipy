"""
In this package, some pre-defined search spaces are provided.
"""
# noinspection PyUnresolvedReferences
from moptipy.version import __version__
from moptipy.spaces.vectorspace import VectorSpace
from moptipy.spaces.intspace import IntSpace
from moptipy.spaces.bitstrings import BitStrings

__all__ = ["BitStrings",
           "IntSpace",
           "VectorSpace"]
