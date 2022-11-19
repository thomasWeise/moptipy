"""Test the end result violin plot."""
from typing import Final

from matplotlib.figure import Figure  # type: ignore

from moptipy.evaluation.plot_end_results import plot_end_results
from moptipy.mock.components import Experiment
from moptipy.mock.end_results import EndResults
from moptipy.utils.plot_utils import create_figure, save_figure
from moptipy.utils.temp import TempDir


def make_end_results_plot(dir_name: str, file_name: str) -> None:
    """
    Make the end result violin plot with random data.

    :param str dir_name: the destination directory
    :param str file_name: the file name base
    """
    exp: Final[Experiment] = Experiment.create(n_instances=4,
                                               n_algorithms=3,
                                               n_runs=20)
    res: Final[EndResults] = EndResults.create(experiment=exp,
                                               max_time_millis=120_000)
    del exp
    fig: Final[Figure] = create_figure(width=7)

    plot_end_results(end_results=res.results, figure=fig)
    save_figure(fig, dir_name=dir_name, file_name=file_name,
                formats=("svg", "pdf"))
    del fig


def test_end_results_plot() -> None:
    """Run the violin plot test."""
    with TempDir.create() as dir_name:
        make_end_results_plot(dir_name, "test")
