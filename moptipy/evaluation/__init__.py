"""Components for parsing and evaluating log files generated by experiments."""

from typing import Final

import moptipy.version
from moptipy.evaluation.axis_ranger import AxisRanger
from moptipy.evaluation.base import PerRunData, MultiRunData
from moptipy.evaluation.ecdf import Ecdf
from moptipy.evaluation.end_results import EndResult
from moptipy.evaluation.end_statistics import EndStatistics
from moptipy.evaluation.ert import Ert, compute_single_ert
from moptipy.evaluation.ertecdf import ErtEcdf
from moptipy.evaluation.log_parser import LogParser, ExperimentParser
from moptipy.evaluation.parse_data import parse_key_values
from moptipy.evaluation.plot_ecdf_impl import plot_ecdf
from moptipy.evaluation.plot_ert_impl import plot_ert
from moptipy.evaluation.plot_progress_impl import plot_progress
from moptipy.evaluation.plot_utils import create_figure, save_figure
from moptipy.evaluation.progress import Progress
from moptipy.evaluation.stat_run import StatRun
from moptipy.evaluation.statistics import Statistics

__version__: Final[str] = moptipy.version.__version__

__all__ = (
    "AxisRanger",
    "compute_single_ert",
    "create_figure",
    "EndResult",
    "EndStatistics",
    "Ecdf",
    "Ert",
    "ErtEcdf",
    "ExperimentParser",
    "LogParser",
    "MultiRunData",
    "parse_key_values",
    "plot_ert",
    "plot_ecdf",
    "plot_progress",
    "PerRunData",
    "Progress",
    "save_figure",
    "Statistics",
    "StatRun")
