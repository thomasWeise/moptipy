"""Components for parsing and evaluating log files generated by experiments."""

from typing import Final

import moptipy.version
from moptipy.evaluation.sections_parser import SectionsParser
from moptipy.evaluation.parse_data import parse_key_values

__version__: Final[str] = moptipy.version.__version__

__all__ = (
    "parse_key_values",
    "SectionsParser", )
