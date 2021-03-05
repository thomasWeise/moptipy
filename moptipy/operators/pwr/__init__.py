"""
Search operators for permutations with repetitions.

These operators can be used in conjunction with the space
:class:`~moptipy.spaces.permutationswr.PermutationsWithRepetitions`.
"""
from typing import Final

import moptipy.version
from moptipy.operators.pwr.op0_shuffle import Op0Shuffle
from moptipy.operators.pwr.op1_swap2 import Op1Swap2

__version__: Final[str] = moptipy.version.__version__

__all__ = (
    "Op0Shuffle",
    "Op1Swap2")
