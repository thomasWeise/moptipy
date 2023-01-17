"""The JSSP-example specific plots."""

from math import inf
from statistics import median
from typing import Any, Callable, Final, Iterable

import moptipy.utils.plot_utils as pu
from moptipy.evaluation.axis_ranger import AxisRanger
from moptipy.evaluation.base import F_NAME_RAW, F_NAME_SCALED, TIME_UNIT_MILLIS
from moptipy.evaluation.end_results import EndResult
from moptipy.evaluation.end_statistics import EndStatistics
from moptipy.evaluation.plot_end_results import plot_end_results
from moptipy.evaluation.plot_end_statistics_over_parameter import (
    plot_end_statistics_over_param,
)
from moptipy.evaluation.plot_progress import plot_progress
from moptipy.evaluation.progress import Progress
from moptipy.evaluation.stat_run import STAT_MEAN_ARITH, StatRun
from moptipy.examples.jssp.plot_gantt_chart import plot_gantt_chart
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
                       algorithm_namer: Callable[[str], str] = lambda x: x,
                       x_label_location: float = 1.0) \
        -> list[Path]:
    """
    Plot a set of end result boxes/violins functions into one chart.

    :param end_results: the iterable of end results
    :param name_base: the basic name
    :param dest_dir: the destination directory
    :param instance_sort_key: the sort key function for instances
    :param algorithm_sort_key: the sort key function for algorithms
    :param algorithm_namer: the name function for algorithms receives an
        algorithm ID and returns an instance name; default=identity function
    :param x_label_location: the location of the label of the x-axis
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

    algorithms: set[str] = set()
    instances: set[str] = set()
    pairs: set[str] = set()
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
    insts: Final[list[str]] = sorted(instances, key=instance_sort_key)
    result: Final[list[Path]] = []

    for lang in Lang.all_langs():
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
                y_label_location=1.0,
                x_label_location=x_label_location,
                algorithm_namer=algorithm_namer)

        result.extend(pu.save_figure(fig=figure,
                                     file_name=lang.filename(name_base),
                                     dir_name=dest_dir))
        del figure

    logger(f"finished plotting chart {name_base}.")
    result.sort()
    return result


def plot_stat_gantt_charts(
        end_results: Iterable[EndResult],
        results_dir: str,
        name_base: str,
        dest_dir: str,
        instance_sort_key: Callable = lambda x: x,
        statistic: Callable[[Iterable[int | float]],
                            int | float] = median) -> list[Path]:
    """
    Plot a set of Gantt charts following a specific statistics.

    :param end_results: the iterable of end results
    :param results_dir: the result directory
    :param name_base: the basic name
    :param dest_dir: the destination directory
    :param instance_sort_key: the sort key function for instances
    :param statistic: the statistic to use
    :returns: the list of generated files
    """
    logger(f"beginning to plot stat chart {name_base}.")
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
    if not callable(statistic):
        raise type_error(statistic, "statistic", call=True)

    results: Final[list[Path]] = []  # the list of generated files

    # gather all the data
    data: Final[dict[str, list[EndResult]]] = {}
    algorithm: str | None = None
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
    instances: Final[list[str]] = sorted(data.keys(), key=instance_sort_key)
    del end_results

    # get the median runs
    stat_runs: list[Path] = []
    results_dir = Path.directory(results_dir)
    for instance in instances:
        runs: list[EndResult] = data[instance]
        runs.sort()
        med = statistic([er.best_f for er in runs])
        best: int | float = inf
        solution: EndResult | None = None
        for er in runs:
            current = abs(er.best_f - med)
            if current < best:
                best = current
                solution = er
        if solution is None:
            raise ValueError(
                f"found no {name_base} end result for instance {instance}.")
        path: Path = solution.path_to_file(results_dir)
        path.enforce_file()
        stat_runs.append(path)
    del data
    del instances
    if len(stat_runs) < 0:
        raise ValueError("empty set of runs?")

    # plot the gantt charts
    for lang in Lang.all_langs():
        lang.set_current()

        figure, plots = pu.create_figure_with_subplots(
            items=len(stat_runs), max_items_per_plot=1, max_cols=2,
            max_rows=4, max_width=8.6, max_height=11.5)

        for plot, start, end, _, _, _ in plots:
            if start != (end - 1):
                raise ValueError(f"{start} != {end} - 1")

            args: dict[str, Any] = {
                "gantt": stat_runs[start],
                "figure": plot,
            }
            if statistic is min:
                args["markers"] = None
            else:
                args["info"] = lambda gantt: \
                    Lang.current().format_str("gantt_info_short", gantt=gantt)
            if len(plots) > 2:
                args["importance_to_font_size_func"] = lambda i: \
                    0.9 * importance_to_font_size(i)

            plot_gantt_chart(**args)

        results.extend(pu.save_figure(fig=figure,
                                      file_name=lang.filename(name_base),
                                      dir_name=dest_dir))

    logger("done plotting stat gantt charts.")
    return results


def plot_progresses(results_dir: str,
                    algorithms: Iterable[str],
                    name_base: str,
                    dest_dir: str,
                    time_unit: str = TIME_UNIT_MILLIS,
                    log_time: bool = True,
                    instance_sort_key: Callable = lambda x: x,
                    algorithm_sort_key: Callable = lambda x: x,
                    x_label_location: float = 0.0,
                    include_runs: bool = False,
                    algorithm_namer: Callable[[str], str] = lambda x: x) \
        -> list[Path]:
    """
    Plot a set of end result boxes/violins functions into one chart.

    :param results_dir: the directory with the log files
    :param algorithms: the set of algorithms to plot together
    :param name_base: the basic name
    :param dest_dir: the destination directory
    :param time_unit: the time unit to plot
    :param log_time: should the time axis be scaled logarithmically?
    :param instance_sort_key: the sort key function for instances
    :param algorithm_sort_key: the sort key function for algorithms
    :param x_label_location: the location of the x-labels
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
    if not isinstance(time_unit, str):
        raise type_error(time_unit, "time_unit", str)
    if not isinstance(log_time, bool):
        raise type_error(log_time, "log_time", bool)
    if not callable(instance_sort_key):
        raise type_error(instance_sort_key, "instance_sort_key", call=True)
    if not callable(algorithm_sort_key):
        raise type_error(algorithm_sort_key, "algorithm_sort_key", call=True)
    if not isinstance(x_label_location, float):
        raise type_error(x_label_location, "x_label_location", float)
    if not isinstance(include_runs, bool):
        raise type_error(include_runs, "include_runs", bool)
    if not callable(algorithm_namer):
        raise type_error(algorithm_namer, "algorithm_namer", call=True)

    # get the data
    spath: Final[Path] = Path.directory(results_dir)
    progresses: Final[list[Progress]] = []
    for algorithm in sorted(algorithms, key=algorithm_sort_key):
        Progress.from_logs(spath.resolve_inside(algorithm), progresses.append,
                           time_unit=time_unit, f_name=F_NAME_RAW)
    if len(progresses) <= 0:
        raise ValueError(f"did not find log files in dir {results_dir!r}.")

    stat_runs: Final[list[Progress | StatRun]] = []
    StatRun.from_progress(progresses, STAT_MEAN_ARITH,
                          stat_runs.append, False, False)
    if len(stat_runs) <= 0:
        raise ValueError(
            f"failed to compile stat runs from dir {results_dir!r}.")
    if include_runs:
        stat_runs.extend(progresses)
    del progresses
    instances: Final[list[str]] = sorted({sr.instance for sr in stat_runs},
                                         key=instance_sort_key)
    if len(instances) <= 0:
        raise ValueError(f"no instances in dir {results_dir!r}.")
    algos: Final[list[str]] = sorted({sr.algorithm for sr in stat_runs},
                                     key=algorithm_sort_key)
    if len(set(algorithms).difference(algos)) > 0:
        raise ValueError(
            f"found the {len(algos)} algorithms {algos}, but expected "
            f"algorithms {algorithms}.")

    results: Final[list[Path]] = []  # the list of generated files

    # plot the progress charts
    for lang in Lang.all_langs():
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
                x_label_location=x_label_location,
                algorithm_namer=algorithm_namer)
            axes = pu.get_axes(plot)
            pu.label_box(axes, inst, x=0.5, y=1)

        results.extend(pu.save_figure(fig=figure,
                                      file_name=lang.filename(name_base),
                                      dir_name=dest_dir))

    logger(f"finished plotting chart {name_base!r}.")
    return results


def plot_end_makespans_over_param(
    end_results: Iterable[EndResult],
    name_base: str,
    dest_dir: str,
    x_getter: Callable[[EndStatistics], int | float],
    algorithm_getter: Callable[[EndStatistics], str | None] =
    lambda es: es.algorithm,
    instance_sort_key: Callable = lambda x: x,
    algorithm_sort_key: Callable = lambda x: x,
    title: str | None = None,
    x_axis: AxisRanger | Callable[[], AxisRanger] = AxisRanger,
    x_label: str | None = None,
    x_label_location: float = 1.0,
    plot_single_instances: bool = True,
    plot_instance_summary: bool = True,
    legend_pos: str = "upper right",
    title_x: float = 0.5,
    y_label_location: float = 1.0) \
        -> list[Path]:
    """
    Plot the performance over a parameter.

    :param end_results: the iterable of end results
    :param name_base: the basic name
    :param dest_dir: the destination directory
    :param title: the optional title
    :param x_getter: the function computing the x-value for each statistics
        object
    :param algorithm_getter: the algorithm getter
    :param instance_sort_key: the sort key function for instances
    :param algorithm_sort_key: the sort key function for algorithms
    :param x_axis: the x_axis ranger
    :param x_label: the x-label
    :param x_label_location: the location of the x-labels
    :param plot_single_instances: shall we plot the values for each single
        instance?
    :param plot_instance_summary: shall we plot the value over all instances?
    :param legend_pos: the legend position
    :param title_x: the title position
    :param y_label_location: the location of the y label
    :returns: the list of generated files
    """
    logger(f"beginning to plot chart {name_base}.")
    if not isinstance(end_results, Iterable):
        raise type_error(end_results, "end_results", Iterable)
    if not isinstance(name_base, str):
        raise type_error(name_base, "name_base", str)
    if not isinstance(dest_dir, str):
        raise type_error(dest_dir, "dest_dir", str)
    if not callable(x_getter):
        raise type_error(x_getter, "x_getter", call=True)
    if not callable(algorithm_getter):
        raise type_error(algorithm_getter, "algorithm_getter", call=True)
    if not callable(instance_sort_key):
        raise type_error(instance_sort_key, "instance_sort_key", call=True)
    if not callable(algorithm_sort_key):
        raise type_error(algorithm_sort_key, "algorithm_sort_key", call=True)
    if not isinstance(x_label_location, float):
        raise type_error(x_label_location, "x_label_location", float)
    if not isinstance(y_label_location, float):
        raise type_error(y_label_location, "y_label_location", float)
    if (title is not None) and (not isinstance(title, str)):
        raise type_error(title, "title", (str, None))
    if not isinstance(plot_single_instances, bool):
        raise type_error(plot_single_instances, "plot_single_instances", bool)
    if not isinstance(plot_instance_summary, bool):
        raise type_error(plot_instance_summary, "plot_instance_summary", bool)
    if not (plot_single_instances or plot_instance_summary):
        raise ValueError("plot_instance_summary and plot_single_instances "
                         "cannot both be False")
    if not isinstance(legend_pos, str):
        raise type_error(legend_pos, "legend_pos", str)
    if not isinstance(title_x, float):
        raise type_error(title_x, "title_x", float)

    logger(f"now plotting end statistics over parameter {title!r}.")

    end_stats: Final[list[EndStatistics]] = []
    if plot_single_instances:
        EndStatistics.from_end_results(end_results, end_stats.append)
    if plot_instance_summary:
        EndStatistics.from_end_results(end_results, end_stats.append,
                                       join_all_instances=True)
    if len(end_stats) <= 0:
        raise ValueError("no end statistics records to plot!")
    result: list[Path] = []

    for lang in Lang.all_langs():
        lang.set_current()
        figure = pu.create_figure(width=5.5)

        axes = plot_end_statistics_over_param(
            data=end_stats, figure=figure,
            algorithm_getter=algorithm_getter,
            x_axis=x_axis,
            x_getter=x_getter,
            x_label=x_label,
            x_label_location=x_label_location,
            algorithm_sort_key=algorithm_sort_key,
            instance_sort_key=instance_sort_key,
            legend_pos=legend_pos,
            y_label_location=y_label_location)
        if title is not None:
            pu.label_box(axes, title, x=title_x, y=1)

        result.extend(pu.save_figure(fig=figure,
                                     file_name=lang.filename(name_base),
                                     dir_name=dest_dir))

    logger(f"done plotting end statistics over parameter {title!r}.")
    return result
