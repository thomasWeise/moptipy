"""The JSSP-example specific plots."""

from math import inf
from statistics import median
from typing import Final, Callable, Iterable, Set, List, Dict, Optional, Union

import moptipy.utils.plot_utils as pu
from moptipy.evaluation.axis_ranger import AxisRanger
from moptipy.evaluation.base import F_NAME_SCALED, TIME_UNIT_MILLIS, F_NAME_RAW
from moptipy.evaluation.end_results import EndResult
from moptipy.evaluation.plot_end_results_impl import plot_end_results
from moptipy.evaluation.plot_progress_impl import plot_progress
from moptipy.evaluation.progress import Progress
from moptipy.evaluation.stat_run import StatRun, STAT_MEAN_ARITH
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
                       algorithm_sort_key: Callable = lambda x: x,
                       algorithm_namer: Callable[[str], str] = lambda x: x) \
        -> List[Path]:
    """
    Plot a set of end result boxes/violins functions into one chart.

    :param end_results: the iterable of end results
    :param name_base: the basic name
    :param dest_dir: the destination directory
    :param instance_sort_key: the sort key function for instances
    :param algorithm_sort_key: the sort key function for algorithms
    :param algorithm_namer: the name function for algorithms receives an
        algorithm ID and returns an instance name; default=identity function
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
                ylabel_location=1,
                algorithm_namer=algorithm_namer)

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

    results: Final[List[Path]] = []  # the list of generated files

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
            max_rows=4, max_width=8.6, max_height=11.5)

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


def plot_progresses(results_dir: str,
                    algorithms: Iterable[str],
                    name_base: str,
                    dest_dir: str,
                    log_time: bool = True,
                    instance_sort_key: Callable = lambda x: x,
                    algorithm_sort_key: Callable = lambda x: x,
                    xlabel_location: float = 0.0,
                    include_runs: bool = False,
                    algorithm_namer: Callable[[str], str] = lambda x: x) \
        -> List[Path]:
    """
    Plot a set of end result boxes/violins functions into one chart.

    :param results_dir: the directory with the log files
    :param algorithms: the set of algorithms to plot together
    :param name_base: the basic name
    :param dest_dir: the destination directory
    :param log_time: should the time axis be scaled logarithmically?
    :param instance_sort_key: the sort key function for instances
    :param algorithm_sort_key: the sort key function for algorithms
    :param xlabel_location: the location of the x-labels
    :param include_runs: should we include the pure runs as well?
    :param algorithm_namer: the name function for algorithms receives an
        algorithm ID and returns an instance name; default=identity function
    :returns: the list of generated files
    """
    logger(f"beginning to plot chart {name_base}.")
    if not isinstance(results_dir, str):
        raise type_error(results_dir, "results_dir", str)
    if not isinstance(algorithms, Iterable):
        raise type_error(algorithms, "algorithms", Iterable)
    if not isinstance(name_base, str):
        raise type_error(name_base, "name_base", str)
    if not isinstance(dest_dir, str):
        raise type_error(dest_dir, "dest_dir", str)
    if not isinstance(log_time, bool):
        raise type_error(log_time, "log_time", bool)
    if not callable(instance_sort_key):
        raise type_error(instance_sort_key, "instance_sort_key", call=True)
    if not callable(algorithm_sort_key):
        raise type_error(algorithm_sort_key, "algorithm_sort_key", call=True)
    if not isinstance(xlabel_location, float):
        raise type_error(xlabel_location, "xlabel_location", float)
    if not isinstance(include_runs, bool):
        raise type_error(include_runs, "include_runs", bool)
    if not callable(algorithm_namer):
        raise type_error(algorithm_namer, "algorithm_namer", call=True)

    # get the data
    spath: Final[Path] = Path.directory(results_dir)
    progresses: Final[List[Progress]] = []
    for algorithm in sorted(algorithms, key=algorithm_sort_key):
        Progress.from_logs(spath.resolve_inside(algorithm), progresses.append,
                           time_unit=TIME_UNIT_MILLIS, f_name=F_NAME_RAW)
    if len(progresses) <= 0:
        raise ValueError(f"did not find log files in dir '{results_dir}'.")

    stat_runs: Final[List[Union[Progress, StatRun]]] = []
    StatRun.from_progress(progresses, STAT_MEAN_ARITH,
                          stat_runs.append, False, False)
    if len(stat_runs) <= 0:
        raise ValueError(
            f"failed to compile stat runs from dir '{results_dir}'.")
    if include_runs:
        stat_runs.extend(progresses)
    del progresses
    instances: Final[List[str]] = sorted({sr.instance for sr in stat_runs},
                                         key=instance_sort_key)
    if len(instances) <= 0:
        raise ValueError(f"no instances in dir '{results_dir}'.")
    algos: Final[List[str]] = sorted({sr.algorithm for sr in stat_runs},
                                     key=algorithm_sort_key)
    if len(set(algorithms).difference(algos)) > 0:
        raise ValueError(
            f"found the {len(algos)} algorithms {algos}, but expected "
            f"algorithms {algorithms}.")

    results: Final[List[Path]] = []  # the list of generated files

    # plot the progress charts
    for lang in Lang.all():
        lang.set_current()
        figure, plots = pu.create_figure_with_subplots(
            items=len(instances), max_items_per_plot=1, max_cols=2,
            max_rows=4, max_width=8.6, max_height=11.5)

        for plot, start, end, _, _, _ in plots:
            if start != (end - 1):
                raise ValueError(f"{start} != {end} - 1")
            inst = instances[start]
            plot_progress(
                progresses=[sr for sr in stat_runs if sr.instance == inst],
                figure=plot,
                x_axis=AxisRanger.for_axis_func(log_scale=log_time),
                importance_to_font_size_func=lambda i:
                0.9 * importance_to_font_size(i),
                algorithm_sort_key=algorithm_sort_key,
                instance_sort_key=instance_sort_key,
                xlabel_location=xlabel_location,
                algorithm_namer=algorithm_namer)
            axes = pu.get_axes(plot)
            pu.label_box(axes, inst, x=0.5, y=1)

        results.extend(pu.save_figure(fig=figure,
                                      file_name=lang.filename(name_base),
                                      dir_name=dest_dir))

    logger(f"finished plotting chart '{name_base}'.")
    return results
