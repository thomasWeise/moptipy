"""The JSSP-example specific plots."""

from math import ceil, floor
from typing import List, Final, Callable, Iterable, Set

from matplotlib.figure import Figure  # type: ignore

from moptipy.evaluation.base import F_NAME_SCALED
from moptipy.evaluation.end_results import EndResult
from moptipy.evaluation.plot_end_results_impl import plot_end_results
import moptipy.evaluation.plot_utils as pu
from moptipy.utils.log import logger
from moptipy.evaluation.lang import Lang


def plot_end_makespans(end_results: Iterable[EndResult],
                       name_base: str,
                       dest_dir: str,
                       instance_sort_key: Callable = lambda x: x,
                       algorithm_sort_key: Callable = lambda x: x) -> None:
    """
    Plot a set of end result boxes/violins functions into one chart.

    :param end_results: the iterable of end results
    :param name_base: the basic name
    :param dest_dir: the destination directory
    :param Callable instance_sort_key: the sort key function for instances
    :param Callable algorithm_sort_key: the sort key function for algorithms
    """
    logger(f"beginning to plot chart {name_base}.")
    algorithms: Set[str] = set()
    instances: Set[str] = set()
    for er in end_results:
        algorithms.add(er.algorithm)
        instances.add(er.instance)

    base_width: Final[float] = 8.6
    base_height: Final[float] = 2.5
    figure: Figure

    n_algos: Final[int] = len(algorithms)
    n_insts: Final[int] = len(instances)
    max_chunk_size: Final[int] = 16
    if n_algos >= max_chunk_size:
        raise ValueError(f"{n_algos} are just too many algorithms...")

    n_insts_per_chunk: int
    n_chunks: int
    insts: List[str] = []
    if (n_algos * n_insts) <= max_chunk_size:
        n_insts_per_chunk = n_insts
        n_chunks = 1
    else:
        n_insts_per_chunk = int(0.5 + floor(max_chunk_size / n_algos))
        n_chunks = int(0.5 + ceil(n_insts / n_insts_per_chunk))
        if n_chunks <= 1:
            raise ValueError("Huh?")
        insts = sorted(instances, key=instance_sort_key, reverse=True)
    logger(f"we will plot the {n_algos} algorithms on {n_insts} in "
           f"{n_chunks} plots with {n_insts_per_chunk} instances per plot.")

    for lang in Lang.all():
        lang.set_current()

        if n_chunks <= 1:
            figure = pu.create_figure(width=base_width, height=base_height)
            plot_end_results(end_results=end_results,
                             figure=figure,
                             dimension=F_NAME_SCALED,
                             instance_sort_key=instance_sort_key,
                             algorithm_sort_key=algorithm_sort_key)
        else:
            figure = pu.create_figure(width=base_width,
                                      height=n_chunks * (1.02 * base_height))
            figures = pu.divide_into_sub_plots(figure=figure,
                                               nrows=n_chunks, ncols=1)
            inst_idx: int = n_insts

            for fig, _, _, _ in figures:
                instances.clear()
                for _ in range(n_insts_per_chunk):
                    if inst_idx > 0:
                        inst_idx -= 1
                        instances.add(insts[inst_idx])
                plot_end_results(end_results=[er for er in end_results
                                              if er.instance in instances],
                                 figure=fig,
                                 dimension=F_NAME_SCALED,
                                 instance_sort_key=instance_sort_key,
                                 algorithm_sort_key=algorithm_sort_key)
        pu.save_figure(fig=figure,
                       file_name=lang.filename(name_base),
                       dir_name=dest_dir)
        del figure

    logger(f"finished plotting chart {name_base}.")
