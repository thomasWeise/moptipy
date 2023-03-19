"""Plot a set of `Progress` or `StatRun` objects into one figure."""
from math import isfinite
from typing import Any, Callable, Final, Iterable

from matplotlib.artist import Artist  # type: ignore
from matplotlib.axes import Axes  # type: ignore
from matplotlib.figure import Figure  # type: ignore

import moptipy.utils.plot_defaults as pd
import moptipy.utils.plot_utils as pu
from moptipy.evaluation.axis_ranger import AxisRanger
from moptipy.evaluation.base import get_algorithm, get_instance, sort_key
from moptipy.evaluation.progress import Progress
from moptipy.evaluation.stat_run import StatRun, get_statistic
from moptipy.evaluation.styler import Styler
from moptipy.utils.lang import Lang
from moptipy.utils.types import type_error


def plot_progress(
        progresses: Iterable[Progress | StatRun],
        figure: Axes | Figure,
        x_axis: AxisRanger | Callable[[str], AxisRanger] =
        AxisRanger.for_axis,
        y_axis: AxisRanger | Callable[[str], AxisRanger] =
        AxisRanger.for_axis,
        legend: bool = True,
        distinct_colors_func: Callable[[int], Any] = pd.distinct_colors,
        distinct_line_dashes_func: Callable[[int], Any] =
        pd.distinct_line_dashes,
        importance_to_line_width_func: Callable[[int], float] =
        pd.importance_to_line_width,
        importance_to_alpha_func: Callable[[int], float] =
        pd.importance_to_alpha,
        importance_to_font_size_func: Callable[[int], float] =
        pd.importance_to_font_size,
        x_grid: bool = True,
        y_grid: bool = True,
        x_label: None | str | Callable[[str], str] = Lang.translate,
        x_label_inside: bool = True,
        x_label_location: float = 0.5,
        y_label: None | str | Callable[[str], str] = Lang.translate,
        y_label_inside: bool = True,
        y_label_location: float = 1.0,
        instance_priority: float = 0.666,
        algorithm_priority: float = 0.333,
        stat_priority: float = 0.0,
        instance_sort_key: Callable[[str], Any] = lambda x: x,
        algorithm_sort_key: Callable[[str], Any] = lambda x: x,
        stat_sort_key: Callable[[str], Any] = lambda x: x,
        color_algorithms_as_fallback_group: bool = True,
        instance_namer: Callable[[str], str] = lambda x: x,
        algorithm_namer: Callable[[str], str] = lambda x: x) -> Axes:
    """
    Plot a set of progress or statistical run lines into one chart.

    :param progresses: the iterable of progresses and statistical runs
    :param figure: the figure to plot in
    :param x_axis: the x_axis ranger
    :param y_axis: the y_axis ranger
    :param legend: should we plot the legend?
    :param distinct_colors_func: the function returning the palette
    :param distinct_line_dashes_func: the function returning the line styles
    :param importance_to_line_width_func: the function converting importance
        values to line widths
    :param importance_to_alpha_func: the function converting importance
        values to alphas
    :param importance_to_font_size_func: the function converting importance
        values to font sizes
    :param x_grid: should we have a grid along the x-axis?
    :param y_grid: should we have a grid along the y-axis?
    :param x_label: a callable returning the label for the x-axis, a label
        string, or `None` if no label should be put
    :param x_label_inside: put the x-axis label inside the plot (so that
        it does not consume additional vertical space)
    :param x_label_location: the location of the x-axis label
    :param y_label: a callable returning the label for the y-axis, a label
        string, or `None` if no label should be put
    :param y_label_inside: put the y-axis label inside the plot (so that
        it does not consume additional horizontal space)
    :param y_label_location: the location of the y-axis label
    :param instance_priority: the style priority for instances
    :param algorithm_priority: the style priority for algorithms
    :param stat_priority: the style priority for statistics
    :param instance_sort_key: the sort key function for instances
    :param algorithm_sort_key: the sort key function for algorithms
    :param stat_sort_key: the sort key function for statistics
    :param color_algorithms_as_fallback_group: if only a single group of data
        was found, use algorithms as group and put them in the legend
    :param instance_namer: the name function for instances receives an
        instance ID and returns an instance name; default=identity function
    :param algorithm_namer: the name function for algorithms receives an
        algorithm ID and returns an algorithm name; default=identity function
    :returns: the axes object to allow you to add further plot elements
    """
    # Before doing anything, let's do some type checking on the parameters.
    # I want to ensure that this function is called correctly before we begin
    # to actually process the data. It is better to fail early than to deliver
    # some incorrect results.
    if not isinstance(progresses, Iterable):
        raise type_error(progresses, "progresses", Iterable)
    if not isinstance(figure, Axes | Figure):
        raise type_error(figure, "figure", (Axes, Figure))
    if not isinstance(legend, bool):
        raise type_error(legend, "legend", bool)
    if not callable(distinct_colors_func):
        raise type_error(
            distinct_colors_func, "distinct_colors_func", call=True)
    if not callable(distinct_colors_func):
        raise type_error(
            distinct_colors_func, "distinct_colors_func", call=True)
    if not callable(distinct_line_dashes_func):
        raise type_error(
            distinct_line_dashes_func, "distinct_line_dashes_func", call=True)
    if not callable(importance_to_line_width_func):
        raise type_error(importance_to_line_width_func,
                         "importance_to_line_width_func", call=True)
    if not callable(importance_to_alpha_func):
        raise type_error(
            importance_to_alpha_func, "importance_to_alpha_func", call=True)
    if not callable(importance_to_font_size_func):
        raise type_error(importance_to_font_size_func,
                         "importance_to_font_size_func", call=True)
    if not isinstance(x_grid, bool):
        raise type_error(x_grid, "x_grid", bool)
    if not isinstance(y_grid, bool):
        raise type_error(y_grid, "y_grid", bool)
    if not ((x_label is None) or callable(x_label)
            or isinstance(x_label, str)):
        raise type_error(x_label, "x_label", (str, None), call=True)
    if not isinstance(x_label_inside, bool):
        raise type_error(x_label_inside, "x_label_inside", bool)
    if not isinstance(x_label_location, float):
        raise type_error(x_label_location, "x_label_location", float)
    if not ((y_label is None) or callable(y_label)
            or isinstance(y_label, str)):
        raise type_error(y_label, "y_label", (str, None), call=True)
    if not isinstance(y_label_inside, bool):
        raise type_error(y_label_inside, "y_label_inside", bool)
    if not isinstance(y_label_location, float):
        raise type_error(y_label_location, "y_label_location", float)
    if not isinstance(instance_priority, float):
        raise type_error(instance_priority, "instance_priority", float)
    if not isfinite(instance_priority):
        raise ValueError(f"instance_priority cannot be {instance_priority}.")
    if not isinstance(algorithm_priority, float):
        raise type_error(algorithm_priority, "algorithm_priority", float)
    if not isfinite(algorithm_priority):
        raise ValueError(f"algorithm_priority cannot be {algorithm_priority}.")
    if not isinstance(stat_priority, float):
        raise type_error(stat_priority, "stat_priority", float)
    if not isfinite(stat_priority):
        raise ValueError(f"stat_priority cannot be {stat_priority}.")
    if not callable(instance_sort_key):
        raise type_error(instance_sort_key, "instance_sort_key", call=True)
    if not callable(algorithm_sort_key):
        raise type_error(algorithm_sort_key, "algorithm_sort_key", call=True)
    if not callable(stat_sort_key):
        raise type_error(stat_sort_key, "stat_sort_key", call=True)
    if not callable(instance_namer):
        raise type_error(instance_namer, "instance_namer", call=True)
    if not callable(algorithm_namer):
        raise type_error(algorithm_namer, "algorithm_namer", call=True)
    if not isinstance(color_algorithms_as_fallback_group, bool):
        raise type_error(color_algorithms_as_fallback_group,
                         "color_algorithms_as_fallback_group", bool)

    # First, we try to find groups of data to plot together in the same
    # color/style. We distinguish progress objects from statistical runs.
    instances: Final[Styler] = Styler(key_func=get_instance,
                                      namer=instance_namer,
                                      none_name=Lang.translate("all_insts"),
                                      priority=instance_priority,
                                      name_sort_function=instance_sort_key)
    algorithms: Final[Styler] = Styler(key_func=get_algorithm,
                                       namer=algorithm_namer,
                                       none_name=Lang.translate("all_algos"),
                                       priority=algorithm_priority,
                                       name_sort_function=algorithm_sort_key)
    statistics: Final[Styler] = Styler(key_func=get_statistic,
                                       none_name=Lang.translate("single_run"),
                                       priority=stat_priority,
                                       name_sort_function=stat_sort_key)
    x_dim: str | None = None
    y_dim: str | None = None
    progress_list: list[Progress] = []
    statrun_list: list[StatRun] = []

    # First pass: find out the statistics, instances, algorithms, and types
    for prg in progresses:
        instances.add(prg)
        algorithms.add(prg)
        statistics.add(prg)
        if isinstance(prg, Progress):
            progress_list.append(prg)
        elif isinstance(prg, StatRun):
            statrun_list.append(prg)
        else:
            raise type_error(prg, "progress plot element",
                             (Progress, StatRun))

        # Validate that we have consistent time and objective units.
        if x_dim is None:
            x_dim = prg.time_unit
        elif x_dim != prg.time_unit:
            raise ValueError(
                f"Time units {x_dim} and {prg.time_unit} do not fit!")

        if y_dim is None:
            y_dim = prg.f_name
        elif y_dim != prg.f_name:
            raise ValueError(
                f"F-units {y_dim} and {prg.f_name} do not fit!")
    del progresses

    if (len(progress_list) + len(statrun_list)) <= 0:
        raise ValueError("Empty input data?")

    if (x_dim is None) or (y_dim is None):
        raise ValueError("Illegal state?")

    instances.finalize()
    algorithms.finalize()
    statistics.finalize()

    # pick the right sorting order
    sf: Callable[[StatRun | Progress], Any] = sort_key
    if (instances.count > 1) and (algorithms.count == 1) \
            and (statistics.count == 1):
        def __x1(r: StatRun | Progress, ssf=instance_sort_key) -> Any:
            return ssf(r.instance)
        sf = __x1
    elif (instances.count == 1) and (algorithms.count > 1) \
            and (statistics.count == 1):
        def __x2(r: StatRun | Progress, ssf=algorithm_sort_key) -> Any:
            return ssf(r.algorithm)
        sf = __x2
    elif (instances.count == 1) and (algorithms.count == 1) \
            and (statistics.count > 1):
        def __x3(r: StatRun | Progress, ssf=stat_sort_key) -> Any:
            return ssf(r.instance)
        sf = __x3
    elif (instances.count > 1) and (algorithms.count > 1):
        def __x4(r: StatRun | Progress, sas=algorithm_sort_key,
                 ias=instance_sort_key,
                 ag=algorithm_priority > instance_priority) \
                -> tuple[Any, Any]:
            k1 = ias(r.instance)
            k2 = sas(r.algorithm)
            return (k2, k1) if ag else (k1, k2)
        sf = __x4

    statrun_list.sort(key=sf)
    progress_list.sort()

    def __set_importance(st: Styler) -> None:
        if st is statistics:
            none = -1
            not_none = 1
        else:
            none = 1
            not_none = 0
        none_lw = importance_to_line_width_func(none)
        not_none_lw = importance_to_line_width_func(not_none)
        st.set_line_width(lambda x: [none_lw if i <= 0 else not_none_lw
                                     for i in range(x)])
        none_a = importance_to_alpha_func(none)
        not_none_a = importance_to_alpha_func(not_none)
        st.set_line_alpha(lambda x: [none_a if i <= 0 else not_none_a
                                     for i in range(x)])

    # determine the style groups
    groups: list[Styler] = []

    no_importance = True
    if instances.count > 1:
        groups.append(instances)
    if algorithms.count > 1:
        groups.append(algorithms)
    add_stat_to_groups = False
    if statistics.count > 1:
        if statistics.has_none and (statistics.count == 2):
            __set_importance(statistics)
            no_importance = False
            add_stat_to_groups = True
        else:
            groups.append(statistics)

    if len(groups) > 0:
        groups.sort()
        groups[0].set_line_color(distinct_colors_func)

        if len(groups) > 1:
            groups[1].set_line_dash(distinct_line_dashes_func)

            if (len(groups) > 2) and no_importance:
                g = groups[2]
                if g.count > 2:
                    raise ValueError(
                        f"Cannot have {g.count} importance values.")
                __set_importance(g)
                no_importance = False
    elif color_algorithms_as_fallback_group:
        algorithms.set_line_color(distinct_colors_func)
        groups.append(algorithms)

    if add_stat_to_groups:
        groups.append(statistics)

    # If we only have <= 2 groups, we can mark None and not-None values with
    # different importance.
    if no_importance and statistics.has_none and (statistics.count > 1):
        __set_importance(statistics)
        no_importance = False
    if no_importance and instances.has_none and (instances.count > 1):
        __set_importance(instances)
        no_importance = False
    if no_importance and algorithms.has_none and (algorithms.count > 1):
        __set_importance(algorithms)

    # we will collect all lines to plot in plot_list
    plot_list: list[dict] = []

    # first we collect all progress object
    for prgs in progress_list:
        style = pd.create_line_style()
        for g in groups:
            g.add_line_style(prgs, style)
        style["x"] = prgs.time
        style["y"] = prgs.f
        plot_list.append(style)
    del progress_list

    # now collect the plot data for the statistics
    for sn in statistics.keys:
        if sn is None:
            continue
        for sr in statrun_list:
            if statistics.key_func(sr) != sn:
                continue

            style = pd.create_line_style()
            for g in groups:
                g.add_line_style(sr, style)
            style["x"] = sr.stat[:, 0]
            style["y"] = sr.stat[:, 1]
            plot_list.append(style)
    del statrun_list

    font_size_0: Final[float] = importance_to_font_size_func(0)

    # set up the graphics area
    axes: Final[Axes] = pu.get_axes(figure)
    axes.tick_params(axis="x", labelsize=font_size_0)
    axes.tick_params(axis="y", labelsize=font_size_0)

    # draw the grid
    if x_grid or y_grid:
        grid_lwd = importance_to_line_width_func(-1)
        if x_grid:
            axes.grid(axis="x", color=pd.GRID_COLOR, linewidth=grid_lwd)
        if y_grid:
            axes.grid(axis="y", color=pd.GRID_COLOR, linewidth=grid_lwd)

    # set up the axis rangers
    if callable(x_axis):
        x_axis = x_axis(x_dim)
    if not isinstance(x_axis, AxisRanger):
        raise type_error(x_axis, "x_axis", AxisRanger)

    if callable(y_axis):
        y_axis = y_axis(y_dim)
    if not isinstance(y_axis, AxisRanger):
        raise type_error(y_axis, "y_axis", AxisRanger)

    # plot the lines
    for line in plot_list:
        axes.step(where="post", **line)
        x_axis.register_array(line["x"])
        y_axis.register_array(line["y"])
    del plot_list

    x_axis.apply(axes, "x")
    y_axis.apply(axes, "y")

    if legend:
        handles: list[Artist] = []

        for g in groups:
            g.add_to_legend(handles.append)
            g.has_style = False

        if instances.has_style:
            instances.add_to_legend(handles.append)
        if algorithms.has_style:
            algorithms.add_to_legend(handles.append)
        if statistics.has_style:
            statistics.add_to_legend(handles.append)

        if len(handles) > 0:
            axes.legend(loc="upper right",
                        handles=handles,
                        labelcolor=[art.color if hasattr(art, "color")
                                    else pd.COLOR_BLACK for art in handles],
                        fontsize=font_size_0)

    pu.label_axes(axes=axes,
                  x_label=x_label(x_dim) if callable(x_label) else x_label,
                  x_label_inside=x_label_inside,
                  x_label_location=x_label_location,
                  y_label=y_label(y_dim) if callable(y_label) else y_label,
                  y_label_inside=y_label_inside,
                  y_label_location=y_label_location,
                  font_size=font_size_0)
    return axes
