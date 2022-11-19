"""Plot the end results over a parameter."""
from math import isfinite
from typing import Any, Callable, Final, Iterable, cast

from matplotlib.artist import Artist  # type: ignore
from matplotlib.axes import Axes  # type: ignore
from matplotlib.figure import Figure, SubplotBase  # type: ignore

import moptipy.utils.plot_defaults as pd
import moptipy.utils.plot_utils as pu
from moptipy.evaluation.axis_ranger import AxisRanger
from moptipy.evaluation.base import F_NAME_SCALED
from moptipy.evaluation.end_statistics import EndStatistics
from moptipy.evaluation.statistics import KEY_MEAN_GEOM
from moptipy.evaluation.styler import Styler
from moptipy.utils.lang import Lang
from moptipy.utils.logger import SCOPE_SEPARATOR
from moptipy.utils.types import type_error


def __make_y_label(y_dim: str) -> str:
    """
    Make the y label.

    :param y_dim: the y dimension
    :returns: the y label
    """
    dotidx: Final[int] = y_dim.find(SCOPE_SEPARATOR)
    if dotidx > 0:
        y_dimension: Final[str] = y_dim[:dotidx]
        y_stat: Final[str] = y_dim[dotidx + 1:]
        return Lang.translate_func(y_stat)(y_dimension)
    return Lang.translate(y_dim)


def __make_y_axis(y_dim: str) -> AxisRanger:
    """
    Make the y axis.

    :param y_dim: the y dimension
    :returns: the y axis
    """
    dotidx: Final[int] = y_dim.find(SCOPE_SEPARATOR)
    if dotidx > 0:
        y_dim = y_dim[:dotidx]
    return AxisRanger.for_axis(y_dim)


def plot_end_statistics_over_param(
        data: Iterable[EndStatistics],
        figure: SubplotBase | Figure,
        x_getter: Callable[[EndStatistics], int | float],
        y_dim: str = f"{F_NAME_SCALED}{SCOPE_SEPARATOR}{KEY_MEAN_GEOM}",
        algorithm_getter: Callable[[EndStatistics], str | None] =
        lambda es: es.algorithm,
        instance_getter: Callable[[EndStatistics], str | None] =
        lambda es: es.instance,
        x_axis: AxisRanger | Callable[[], AxisRanger] = AxisRanger,
        y_axis: AxisRanger | Callable[[str], AxisRanger] =
        __make_y_axis,
        legend: bool = True,
        legend_pos: str = "upper right",
        distinct_colors_func: Callable[[int], Any] = pd.distinct_colors,
        distinct_line_dashes_func: Callable[[int], Any] =
        pd.distinct_line_dashes,
        importance_to_line_width_func: Callable[[int], float] =
        pd.importance_to_line_width,
        importance_to_font_size_func: Callable[[int], float] =
        pd.importance_to_font_size,
        x_grid: bool = True,
        y_grid: bool = True,
        x_label: str | None = None,
        x_label_inside: bool = True,
        x_label_location: float = 0.5,
        y_label: None | str | Callable[[str], str] = __make_y_label,
        y_label_inside: bool = True,
        y_label_location: float = 1.0,
        inst_priority: float = 0.666,
        algo_priority: float = 0.333,
        stat_priority: float = 0.0,
        instance_sort_key: Callable[[str], Any] = lambda x: x,
        algorithm_sort_key: Callable[[str], Any] = lambda x: x,
        instance_namer: Callable[[str], str] = lambda x: x,
        algorithm_namer: Callable[[str], str] = lambda x: x,
        stat_sort_key: Callable[[str], str] = lambda x: x,
        color_algorithms_as_fallback_group: bool = True) -> Axes:
    """
    Plot a series of end result statistics over a parameter.

    :param data: the iterable of EndStatistics
    :param figure: the figure to plot in
    :param x_getter: the function computing the x-value for each statistics
        object
    :param y_dim: the dimension to be plotted along the y-axis
    :param algorithm_getter: the algorithm getter
    :param instance_getter: the instance getter
    :param x_axis: the x_axis ranger
    :param y_axis: the y_axis ranger
    :param legend: should we plot the legend?
    :param legend_pos: the legend position
    :param distinct_colors_func: the function returning the palette
    :param distinct_line_dashes_func: the function returning the line styles
    :param importance_to_line_width_func: the function converting importance
        values to line widths
    :param importance_to_font_size_func: the function converting importance
        values to font sizes
    :param x_grid: should we have a grid along the x-axis?
    :param y_grid: should we have a grid along the y-axis?
    :param x_label: the label for the x-axi or `None` if no label should be put
    :param x_label_inside: put the x-axis label inside the plot (so that
        it does not consume additional vertical space)
    :param x_label_location: the location of the x-axis label
    :param y_label: a callable returning the label for the y-axis, a label
        string, or `None` if no label should be put
    :param y_label_inside: put the y-axis label inside the plot (so that
        it does not consume additional horizontal space)
    :param y_label_location: the location of the y-axis label
    :param inst_priority: the style priority for instances
    :param algo_priority: the style priority for algorithms
    :param stat_priority: the style priority for statistics
    :param instance_sort_key: the sort key function for instances
    :param algorithm_sort_key: the sort key function for algorithms
    :param instance_namer: the name function for instances receives an
        instance ID and returns an instance name; default=identity function
    :param algorithm_namer: the name function for algorithms receives an
        algorithm ID and returns an instance name; default=identity function
    :param stat_sort_key: the sort key function for statistics
    :param color_algorithms_as_fallback_group: if only a single group of data
        was found, use algorithms as group and put them in the legend
    :returns: the axes object to allow you to add further plot elements
    """
    # Before doing anything, let's do some type checking on the parameters.
    # I want to ensure that this function is called correctly before we begin
    # to actually process the data. It is better to fail early than to deliver
    # some incorrect results.
    if not isinstance(data, Iterable):
        raise type_error(data, "data", Iterable)
    if not isinstance(figure, (SubplotBase, Figure)):
        raise type_error(figure, "figure", (SubplotBase, Figure))
    if not callable(x_getter):
        raise type_error(x_getter, "x_getter", call=True)
    if not isinstance(y_dim, str):
        raise type_error(y_dim, "y_dim", str)
    if len(y_dim) <= 0:
        raise ValueError(f"invalid y-dimension '{y_dim}'")
    if not callable(instance_getter):
        raise type_error(instance_getter, "instance_getter", call=True)
    if not callable(algorithm_getter):
        raise type_error(algorithm_getter, "algorithm_getter", call=True)
    if not isinstance(legend, bool):
        raise type_error(legend, "legend", bool)
    if not isinstance(legend_pos, str):
        raise type_error(legend_pos, "legend_pos", str)
    if not callable(distinct_colors_func):
        raise type_error(
            distinct_colors_func, "distinct_colors_func", call=True)
    if not callable(distinct_colors_func):
        raise type_error(
            distinct_colors_func, "distinct_colors_func", call=True)
    if not callable(distinct_line_dashes_func):
        raise type_error(
            distinct_line_dashes_func, "distinct_line_dashes_func", call=True)
    if not callable(importance_to_font_size_func):
        raise type_error(importance_to_font_size_func,
                         "importance_to_font_size_func", call=True)
    if not isinstance(x_grid, bool):
        raise type_error(x_grid, "x_grid", bool)
    if not isinstance(y_grid, bool):
        raise type_error(y_grid, "y_grid", bool)
    if not ((x_label is None) or isinstance(x_label, str)):
        raise type_error(x_label, "x_label", (str, None))
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
    if not isinstance(inst_priority, float):
        raise type_error(inst_priority, "inst_priority", float)
    if not isfinite(inst_priority):
        raise ValueError(f"inst_priority cannot be {inst_priority}.")
    if not isinstance(algo_priority, float):
        raise type_error(algo_priority, "algo_priority", float)
    if not isfinite(algo_priority):
        raise ValueError(f"algo_priority cannot be {algo_priority}.")
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

    # the getter for the dimension value
    y_getter: Final[Callable[[EndStatistics], int | float]] \
        = cast(Callable[[EndStatistics], int | float],
               EndStatistics.getter(y_dim))
    if not callable(y_getter):
        raise type_error(y_getter, "y-getter", call=True)

    # set up the axis rangers
    if callable(x_axis):
        x_axis = x_axis()
    if not isinstance(x_axis, AxisRanger):
        raise type_error(x_axis, "x_axis", AxisRanger)

    if callable(y_axis):
        y_axis = y_axis(y_dim)
    if not isinstance(y_axis, AxisRanger):
        raise type_error(y_axis, "y_axis", AxisRanger)

    # First, we try to find groups of data to plot together in the same
    # color/style. We distinguish progress objects from statistical runs.
    instances: Final[Styler] = Styler(
        none_name=Lang.translate("all_insts"),
        priority=inst_priority,
        namer=instance_namer,
        name_sort_function=instance_sort_key)
    algorithms: Final[Styler] = Styler(
        none_name=Lang.translate("all_algos"),
        namer=algorithm_namer,
        priority=algo_priority, name_sort_function=algorithm_sort_key)

    # we now extract the data: x -> algo -> inst -> y
    dataset: Final[dict[str | None, dict[
        str | None, dict[int | float, int | float]]]] = {}
    for endstat in data:
        if not isinstance(endstat, EndStatistics):
            raise type_error(endstat, "element in data", EndStatistics)
        x_value = x_getter(endstat)
        if not isinstance(x_value, (int, float)):
            raise type_error(x_value, "x-value", (int, float))
        _algo = algorithm_getter(endstat)
        if not ((_algo is None) or isinstance(_algo, str)):
            raise type_error(_algo, "algorithm name", None, call=True)
        _inst = instance_getter(endstat)
        if not ((_inst is None) or isinstance(_inst, str)):
            raise type_error(_algo, "instance name", None, call=True)
        y_value = y_getter(endstat)
        if not isinstance(y_value, (int, float)):
            raise type_error(y_value, "y-value", (int, float))
        if _algo in dataset:
            _dataset = dataset[_algo]
        else:
            dataset[_algo] = _dataset = {}
        if _inst in _dataset:
            __dataset = _dataset[_inst]
        else:
            _dataset[_inst] = __dataset = {}
        if x_value in __dataset:
            raise ValueError(
                f"combination x={x_value}, algo='{_algo}', inst='{_inst}' "
                f"already known as value {__dataset[x_value]}, cannot assign "
                f"value {y_value}.")
        __dataset[x_value] = y_value
        x_axis.register_value(x_value)
        y_axis.register_value(y_value)
        algorithms.add(_algo)
        instances.add(_inst)
    del data, y_getter, x_getter, x_value, y_value

    if len(dataset) <= 0:
        raise ValueError("no data found?")

    def __set_importance(st: Styler) -> None:
        none = 1
        not_none = 0
        none_lw = importance_to_line_width_func(none)
        not_none_lw = importance_to_line_width_func(not_none)
        st.set_line_width(lambda p: [none_lw if i <= 0 else not_none_lw
                                     for i in range(p)])

    # determine the style groups
    groups: list[Styler] = []
    instances.finalize()
    algorithms.finalize()

    if instances.count > 1:
        groups.append(instances)
    if algorithms.count > 1:
        groups.append(algorithms)

    if len(groups) > 0:
        groups.sort()
        groups[0].set_line_color(distinct_colors_func)

        if len(groups) > 1:
            groups[1].set_line_dash(distinct_line_dashes_func)
    elif color_algorithms_as_fallback_group:
        algorithms.set_line_color(distinct_colors_func)
        groups.append(algorithms)

    # If we only have <= 2 groups, we can mark None and not-None values with
    # different importance.
    if instances.has_none and (instances.count > 1):
        __set_importance(instances)
    elif algorithms.has_none and (algorithms.count > 1):
        __set_importance(algorithms)

    # we will collect all lines to plot in plot_list
    plot_list: list[dict] = []
    for algo in algorithms.keys:
        _dataset = dataset[algo]
        for inst in instances.keys:
            if inst not in _dataset:
                raise ValueError(f"instance '{inst}' not in dataset"
                                 f" for algorithm '{algo}'.")
            __dataset = _dataset[inst]
            style = pd.create_line_style()
            style["x"] = x_vals = sorted(__dataset.keys())
            style["y"] = [__dataset[x] for x in x_vals]
            for g in groups:
                g.add_line_style(inst if g is instances else algo, style)
            plot_list.append(style)
    del dataset, _dataset, __dataset

    # now we have all data, let's move to the actual plotting
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

    # plot the lines
    for line in plot_list:
        axes.step(where="post", **line)
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

        if len(handles) > 0:
            axes.legend(loc=legend_pos,
                        handles=handles,
                        labelcolor=[art.color if hasattr(art, "color")
                                    else pd.COLOR_BLACK for art in handles],
                        fontsize=font_size_0)

    pu.label_axes(axes=axes,
                  x_label=x_label,
                  x_label_inside=x_label_inside,
                  x_label_location=x_label_location,
                  y_label=y_label(y_dim) if callable(y_label) else y_label,
                  y_label_inside=y_label_inside,
                  y_label_location=y_label_location,
                  font_size=font_size_0)
    return axes
