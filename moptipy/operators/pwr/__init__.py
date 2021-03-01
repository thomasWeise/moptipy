"""
In this package, we provide implementations for search operators to be used in
conjunction with the space
:class:`~moptipy.spaces.permutationswr.PermutationsWithRepetitions`.
"""
# noinspection PyUnresolvedReferences
from moptipy.version import __version__
from moptipy.operators.pwr.op0 import Op0
from moptipy.operators.pwr.op1_swap2 import Op1Swap2

__all__ = ("Op0",
           "Op1Swap2")
