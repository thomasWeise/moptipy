"""
In this package, we provide implementations for search operators to be used in
conjunction with the space
:class:`~moptipy.spaces.permutationswr.PermutationsWithRepetitions`.
"""
from typing import Final, Tuple

import moptipy.version
from moptipy.operators.pwr.op0_shuffle import Op0Shuffle
from moptipy.operators.pwr.op1_swap2 import Op1Swap2

__version__: Final[str] = moptipy.version.__version__

__all__: Final[Tuple[str, ...]] = (
    "Op0Shuffle",
    "Op1Swap2")
