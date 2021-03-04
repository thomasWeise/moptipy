"""
In this package, we provide implementations for search operators to be used in
conjunction with the space
:class:`~moptipy.spaces.permutationswr.PermutationsWithRepetitions`.
"""
from moptipy.operators.pwr.op0_shuffle import Op0Shuffle
from moptipy.operators.pwr.op1_swap2 import Op1Swap2
# noinspection PyUnresolvedReferences
from moptipy.version import __version__

__all__ = ("Op0Shuffle",
           "Op1Swap2")
