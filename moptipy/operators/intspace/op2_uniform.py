"""The bit-string based uniform crossover operator used for intspaces."""

from typing import Final

from moptipy.operators.bitstrings.op2_uniform import Op2Uniform as __Op2Uni

#: The uniform crossover operator for integer spaces is the same as for
#: bitstrings.
Op2Uniform: Final[type[__Op2Uni]] = __Op2Uni
