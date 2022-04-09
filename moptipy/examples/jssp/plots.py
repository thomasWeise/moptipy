"""The JSSP-example specific plots."""

from typing import Final, Callable, Iterable, Set, List

import moptipy.utils.plot_utils as pu
from moptipy.evaluation.base import F_NAME_SCALED
from moptipy.evaluation.end_results import EndResult
from moptipy.evaluation.lang import Lang
from moptipy.evaluation.plot_end_results_impl import plot_end_results
from moptipy.utils.console import logger


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
    :param instance_sort_key: the sort key function for instances
    :param algorithm_sort_key: the sort key function for algorithms
    """
    logger(f"beginning to plot chart {name_base}.")
    algorithms: Set[str] = set()
    instances: Set[str] = set()
    pairs: Set[str] = set()
    for er in end_results:
        algorithms.add(er.algorithm)
        instances.add(er.instance)
        pairs.add(f"{er.algorithm}+{er.instance}")

    n_algos: Final[int] = len(algorithms)
    n_insts: Final[int] = len(instances)
    n_pairs: Final[int] = len(pairs)
    if n_pairs != (n_algos * n_insts):
        raise ValueError(
            f"found {n_algos} algorithms and {n_insts} instances, "
            f"but only {n_pairs} algorithm-instance pairs!")

    if n_algos >= 16:
        raise ValueError(f"{n_algos} are just too many algorithms...")
    max_insts: Final[int] = 16 // n_algos
    insts: Final[List[str]] = sorted(instances, key=instance_sort_key)

    for lang in Lang.all():
        lang.set_current()
        figure, plots = pu.create_figure_with_subplots(
            items=n_insts, max_items_per_plot=max_insts, max_rows=5,
            max_cols=1, max_width=8.6, default_height_per_row=2.5)

        for plot, start_inst, end_inst, _, _, _ in plots:
            instances.clear()
            instances.update(insts[start_inst:end_inst])
            plot_end_results(end_results=[er for er in end_results
                                          if er.instance in instances],
                             figure=plot,
                             dimension=F_NAME_SCALED,
                             instance_sort_key=instance_sort_key,
                             algorithm_sort_key=algorithm_sort_key)

        pu.save_figure(fig=figure,
                       file_name=lang.filename(name_base),
                       dir_name=dest_dir)
        del figure

    logger(f"finished plotting chart {name_base}.")
