"""Plot a set of ERT objects into one figure."""
from typing import Any, Callable, Final, Iterable, cast

import numpy as np
from matplotlib.artist import Artist  # type: ignore
from matplotlib.axes import Axes  # type: ignore
from matplotlib.figure import Figure, SubplotBase  # type: ignore

import moptipy.utils.plot_defaults as pd
import moptipy.utils.plot_utils as pu
from moptipy.evaluation.axis_ranger import AxisRanger
from moptipy.evaluation.base import get_algorithm, get_instance, sort_key
from moptipy.evaluation.ert import Ert
from moptipy.evaluation.styler import Styler
from moptipy.utils.lang import Lang
from moptipy.utils.types import type_error


def plot_ert(erts: Iterable[Ert],
             figure: SubplotBase | Figure,
             x_axis: AxisRanger | Callable[[str], AxisRanger]
             = AxisRanger.for_axis,
             y_axis: AxisRanger | Callable[[str], AxisRanger]
             = AxisRanger.for_axis,
             legend: bool = True,
             distinct_colors_func: Callable[[int], Any] =
             pd.distinct_colors,
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
             y_label: None | str | Callable[[str], str] =
             Lang.translate_func("ERT"),
             y_label_inside: bool = True,
             inst_priority: float = 0.666,
             algo_priority: float = 0.333) -> Axes:
    """
    Plot a set of Ert functions into one chart.

    :param erts: the iterable of Ert functions
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
    :param y_label: a callable returning the label for the y-axis, a label
        string, or `None` if no label should be put
    :param y_label_inside: put the y-axis label inside the plot (so that
        it does not consume additional horizontal space)
    :param inst_priority: the style priority for instances
    :param algo_priority: the style priority for algorithms
    :returns: the axes object to allow you to add further plot elements
    """
    # Before doing anything, let's do some type checking on the parameters.
    # I want to ensure that this function is called correctly before we begin
    # to actually process the data. It is better to fail early than to deliver
    # some incorrect results.
    if not isinstance(erts, Iterable):
        raise type_error(erts, "erts", Iterable)
    if not isinstance(figure, (SubplotBase, Figure)):
        raise type_error(figure, "figure", (SubplotBase, Figure))
    if not isinstance(legend, bool):
        raise type_error(legend, "legend", bool)
    if not callable(distinct_colors_func):
        raise type_error(
            distinct_colors_func, "distinct_colors_func", call=True)
    if not callable(distinct_line_dashes_func):
        raise type_error(
            distinct_line_dashes_func, "distinct_line_dashes_func", call=True)
    if not callable(distinct_line_dashes_func):
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
    if not ((y_label is None) or callable(y_label)
            or isinstance(y_label, str)):
        raise type_error(y_label, "y_label", (str, None), call=True)
    if not isinstance(y_label_inside, bool):
        raise type_error(y_label_inside, "y_label_inside", bool)
    if not isinstance(inst_priority, float):
        raise type_error(inst_priority, "inst_priority", float)
    if not isinstance(algo_priority, float):
        raise type_error(algo_priority, "algo_priority", float)

    # First, we try to find groups of data to plot together in the same
    # color/style.
    instances: Final[Styler] = Styler(
        key_func=get_instance,
        none_name=Lang.translate("all_insts"),
        priority=inst_priority)
    algorithms: Final[Styler] = Styler(
        key_func=get_algorithm,
        none_name=Lang.translate("all_algos"),
        priority=algo_priority)
    x_dim: str | None = None
    y_dim: str | None = None
    source: list[Ert] = cast(list[Ert], erts) if isinstance(erts, list) \
        else list(erts)
    del erts

    # First pass: find out the instances and algorithms
    for ert in source:
        if not isinstance(ert, Ert):
            raise type_error(ert, "ert data source", Ert)
        instances.add(ert)
        algorithms.add(ert)

        # Validate that we have consistent time and objective units.
        if x_dim is None:
            x_dim = ert.f_name
        elif x_dim != ert.f_name:
            raise ValueError(
                f"F-units {x_dim} and {ert.f_name} do not fit!")

        if y_dim is None:
            y_dim = ert.time_unit
        elif y_dim != ert.time_unit:
            raise ValueError(
                f"Time units {y_dim} and {ert.time_unit} do not fit!")

    if (x_dim is None) or (y_dim is None) or (len(source) <= 0):
        raise ValueError("Illegal state?")

    source.sort(key=sort_key)

    def __set_importance(st: Styler) -> None:
        none = 1
        not_none = 0
        none_lw = importance_to_line_width_func(none)
        not_none_lw = importance_to_line_width_func(not_none)
        st.set_line_width(lambda p: [none_lw if i <= 0 else not_none_lw
                                     for i in range(p)])
        none_a = importance_to_alpha_func(none)
        not_none_a = importance_to_alpha_func(not_none)
        st.set_line_alpha(lambda p: [none_a if i <= 0 else not_none_a
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

    # If we only have <= 2 groups, we can mark None and not-None values with
    # different importance.
    if instances.has_none and (instances.count > 1):
        __set_importance(instances)
    elif algorithms.has_none and (algorithms.count > 1):
        __set_importance(algorithms)

    # we will collect all lines to plot in plot_list
    plot_list: list[dict] = []

    # set up the axis rangers
    if callable(x_axis):
        x_axis = x_axis(x_dim)
    if not isinstance(x_axis, AxisRanger):
        raise type_error(x_axis, "x_axis", AxisRanger)

    if callable(y_axis):
        y_axis = y_axis(y_dim)
    if not isinstance(y_axis, AxisRanger):
        raise type_error(y_axis, "y_axis", AxisRanger)

    # first we collect all progress object
    for ert in source:
        style = pd.create_line_style()
        for g in groups:
            g.add_line_style(ert, style)
        x = ert.ert[:, 0]
        style["x"] = x
        x_axis.register_array(x)
        y = ert.ert[:, 1]
        y_axis.register_array(y)
        style["y"] = y
        plot_list.append(style)
    del source

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

    max_y = y_axis.get_pinf_replacement()

    # plot the lines
    for line in plot_list:
        y = line["y"]
        if np.isposinf(y[0]):
            y = y.copy()
            y[0] = max_y
            line["y"] = y
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

        axes.legend(loc="upper right",
                    handles=handles,
                    labelcolor=[art.color if hasattr(art, "color")
                                else pd.COLOR_BLACK for art in handles],
                    fontsize=font_size_0)

    pu.label_axes(axes=axes,
                  x_label=x_label(x_dim) if callable(x_label) else x_label,
                  x_label_inside=x_label_inside,
                  x_label_location=1,
                  y_label=y_label(y_dim) if callable(y_label) else y_label,
                  y_label_inside=y_label_inside,
                  y_label_location=0,
                  font_size=font_size_0)
    return axes
