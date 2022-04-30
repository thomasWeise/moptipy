"""The JSSP-example specific plots."""

from math import inf
from statistics import median
from typing import Final, Callable, Iterable, Set, List, Dict, Optional, Union

import moptipy.utils.plot_utils as pu
from moptipy.evaluation.base import F_NAME_SCALED
from moptipy.evaluation.end_results import EndResult
from moptipy.evaluation.plot_end_results_impl import plot_end_results
from moptipy.examples.jssp.plot_gantt_chart_impl import plot_gantt_chart
from moptipy.utils.console import logger
from moptipy.utils.lang import Lang
from moptipy.utils.path import Path
from moptipy.utils.plot_defaults import importance_to_font_size
from moptipy.utils.types import type_error


def plot_end_makespans(end_results: Iterable[EndResult],
                       name_base: str,
                       dest_dir: str,
                       instance_sort_key: Callable = lambda x: x,
                       algorithm_sort_key: Callable = lambda x: x) \
        -> List[Path]:
    """
    Plot a set of end result boxes/violins functions into one chart.

    :param end_results: the iterable of end results
    :param name_base: the basic name
    :param dest_dir: the destination directory
    :param instance_sort_key: the sort key function for instances
    :param algorithm_sort_key: the sort key function for algorithms
    :returns: the list of generated files
    """
    logger(f"beginning to plot chart {name_base}.")
    if not isinstance(end_results, Iterable):
        raise type_error(end_results, "end_results", Iterable)
    if not isinstance(name_base, str):
        raise type_error(name_base, "name_base", str)
    if not isinstance(dest_dir, str):
        raise type_error(dest_dir, "dest_dir", str)
    if not callable(instance_sort_key):
        raise type_error(instance_sort_key, "instance_sort_key", call=True)
    if not callable(algorithm_sort_key):
        raise type_error(algorithm_sort_key, "algorithm_sort_key", call=True)

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
    result: Final[List[Path]] = []

    for lang in Lang.all():
        lang.set_current()
        figure, plots = pu.create_figure_with_subplots(
            items=n_insts, max_items_per_plot=max_insts, max_rows=5,
            max_cols=1, max_width=8.6, default_height_per_row=2.5)

        for plot, start_inst, end_inst, _, _, _ in plots:
            instances.clear()
            instances.update(insts[start_inst:end_inst])
            plot_end_results(
                end_results=[er for er in end_results
                             if er.instance in instances],
                figure=plot,
                dimension=F_NAME_SCALED,
                instance_sort_key=instance_sort_key,
                algorithm_sort_key=algorithm_sort_key,
                ylabel_location=1)

        result.extend(pu.save_figure(fig=figure,
                                     file_name=lang.filename(name_base),
                                     dir_name=dest_dir))
        del figure

    logger(f"finished plotting chart {name_base}.")
    result.sort()
    return result


def plot_median_gantt_charts(
        end_results: Iterable[EndResult],
        results_dir: str,
        name_base: str,
        dest_dir: str,
        instance_sort_key: Callable = lambda x: x) -> List[Path]:
    """
    Plot a set of end result boxes/violins functions into one chart.

    :param end_results: the iterable of end results
    :param results_dir: the result directory
    :param name_base: the basic name
    :param dest_dir: the destination directory
    :param instance_sort_key: the sort key function for instances
    :returns: the list of generated files
    """
    logger(f"beginning to plot median chart {name_base}.")
    if not isinstance(end_results, Iterable):
        raise type_error(end_results, "end_results", Iterable)
    if not isinstance(results_dir, str):
        raise type_error(results_dir, "results_dir", str)
    if not isinstance(name_base, str):
        raise type_error(name_base, "name_base", str)
    if not isinstance(dest_dir, str):
        raise type_error(dest_dir, "dest_dir", str)
    if not callable(instance_sort_key):
        raise type_error(instance_sort_key, "instance_sort_key", call=True)

    results: Final[List[Path]] = []

    # gather all the data
    data: Final[Dict[str, List[EndResult]]] = {}
    algorithm: Optional[str] = None
    for er in end_results:
        if algorithm is None:
            algorithm = er.algorithm
        elif algorithm != er.algorithm:
            raise ValueError(
                f"found two algorithms: {algorithm} and {er.algorithm}!")
        if er.instance in data:
            data[er.instance].append(er)
        else:
            data[er.instance] = [er]
    if algorithm is None:
        raise ValueError("Did not encounter any algorithm?")
    instances: Final[List[str]] = sorted(data.keys(), key=instance_sort_key)
    del end_results

    # get the median runs
    median_runs: List[Path] = []
    results_dir = Path.directory(results_dir)
    for instance in instances:
        runs: List[EndResult] = data[instance]
        runs.sort()
        med = median([er.best_f for er in runs])
        best: Union[int, float] = inf
        solution: Optional[EndResult] = None
        for er in runs:
            current = abs(er.best_f - med)
            if current < best:
                best = current
                solution = er
        if solution is None:
            raise ValueError(
                f"found no median end result for instance {instance}.")
        path: Path = solution.path_to_file(results_dir)
        path.enforce_file()
        median_runs.append(path)
    del data
    del instances

    # plot the gantt charts
    for lang in Lang.all():
        lang.set_current()
        figure, plots = pu.create_figure_with_subplots(
            items=len(median_runs), max_items_per_plot=1, max_cols=2,
            max_rows=4, max_width=8.6, max_height=11)

        for plot, start, end, _, _, _ in plots:
            if start != (end - 1):
                raise ValueError(f"{start} != {end} - 1")
            plot_gantt_chart(
                gantt=median_runs[start], figure=plot,
                importance_to_font_size_func=lambda i:
                0.9 * importance_to_font_size(i),
                info=lambda gantt:
                Lang.current().format("gantt_info_short", gantt=gantt))

        results.extend(pu.save_figure(fig=figure,
                                      file_name=lang.filename(name_base),
                                      dir_name=dest_dir))

    logger("done plotting median gantt charts.")
    return results
