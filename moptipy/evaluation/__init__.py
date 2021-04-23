"""Components for parsing and evaluating log files generated by experiments."""

from typing import Final

import moptipy.version
from moptipy.evaluation.axis_ranger import AxisRanger
from moptipy.evaluation.base import Setup, PerRunData, MultiRunData
from moptipy.evaluation.end_results import EndResult
from moptipy.evaluation.end_statistics import EndStatistics
from moptipy.evaluation.log_parser import LogParser, ExperimentParser
from moptipy.evaluation.parse_data import parse_key_values
from moptipy.evaluation.progress import Progress
from moptipy.evaluation.stat_run import StatRun
from moptipy.evaluation.statistics import Statistics
from moptipy.evaluation.plot_progress_impl import plot_progress
from moptipy.evaluation.plot_utils import create_figure, save_figure
from moptipy.evaluation.ert import Ert, compute_single_ert

__version__: Final[str] = moptipy.version.__version__

__all__ = (
    "AxisRanger",
    "compute_single_ert",
    "create_figure",
    "EndResult",
    "EndStatistics",
    "Ert",
    "ExperimentParser",
    "LogParser",
    "MultiRunData",
    "parse_key_values",
    "plot_progress",
    "PerRunData",
    "Progress",
    "save_figure",
    "Setup",
    "Statistics",
    "StatRun")
