"""Plot a set of ERT-ECDF objects into one figure."""
from typing import List, Dict, Final, Callable, Iterable, Union, \
    Optional, cast, Set

import numpy as np
from matplotlib.artist import Artist  # type: ignore
from matplotlib.axes import Axes  # type: ignore
from matplotlib.figure import Figure, SubplotBase  # type: ignore

import moptipy.evaluation.plot_defaults as pd
import moptipy.evaluation.plot_utils as pu
from moptipy.evaluation.axis_ranger import AxisRanger
from moptipy.evaluation.base import get_algorithm, sort_key
from moptipy.evaluation.ecdf import Ecdf, get_goal, goal_to_str
from moptipy.evaluation.styler import Styler


def ecdf_x_axis_label(x: List[str]) -> str:
    """
    Compute the proper label for the x-axis for the ECDF plot.

    :param str x: the collected x-axis labels
    :return: the label for the x-axis
    :rtype: str
    """
    return x[0]


def ecdf_y_axis_label(y: str) -> str:
    """
    Compute the proper label for the y-axis for the ERT-ECDF.

    :param str y: the y-axis (time) type
    :return: the label for the y-axis
    :rtype: str
    """
    return f"ecdf\u2009[{pd.default_axis_label(y)}]"


def plot_ecdf(ecdfs: Iterable[Ecdf],
              figure: Union[SubplotBase, Figure],
              x_axis: Union[AxisRanger, Callable] = AxisRanger.for_axis,
              y_axis: Union[AxisRanger, Callable] = AxisRanger.for_axis,
              legend: bool = True,
              distinct_colors_func: Callable = pd.distinct_colors,
              distinct_line_dashes_func: Callable =
              pd.distinct_line_dashes,
              importance_to_line_width_func: Callable =
              pd.importance_to_line_width,
              importance_to_alpha_func: Callable =
              pd.importance_to_alpha,
              importance_to_font_size_func: Callable =
              pd.importance_to_font_size,
              xgrid: bool = True,
              ygrid: bool = True,
              xlabel: Union[None, str, Callable] = ecdf_x_axis_label,
              xlabel_inside: bool = True,
              ylabel: Union[None, str, Callable] = ecdf_y_axis_label,
              ylabel_inside: bool = True,
              algo_priority: float = 5.0,
              goal_priority: float = 0.333) -> None:
    """
    Plot a set of Ecdf functions into one chart.

    :param ecdfs: the iterable of ECDF functions
    :param Union[SubplotBase, Figure] figure: the figure to plot in
    :param Union[moptipy.evaluation.AxisRanger, Callable] x_axis: the x_axis
    :param Union[moptipy.evaluation.AxisRanger, Callable] y_axis: the y_axis
    :param bool legend: should we plot the legend?
    :param Callable distinct_colors_func: the function returning the palette
    :param Callable distinct_line_dashes_func: the function returning the line
        styles
    :param Callable importance_to_line_width_func: the function converting
        importance values to line widths
    :param Callable importance_to_alpha_func: the function converting
        importance values to alphas
    :param Callable importance_to_font_size_func: the function converting
        importance values to font sizes
    :param bool xgrid: should we have a grid along the x-axis?
    :param bool ygrid: should we have a grid along the y-axis
    :param Union[None,str,Callable] xlabel: a callable returning the label for
        the x-axis, a label string, or `None` if no label should be put
    :param bool xlabel_inside: put the x-axis label inside the plot (so that
        it does not consume additional vertical space)
    :param Union[None,str,Callable] ylabel: a callable returning the label for
        the y-axis, a label string, or `None` if no label should be put
    :param bool ylabel_inside: put the y-axis label inside the plot (so that
        it does not consume additional horizontal space)
    :param float algo_priority: the style priority for algorithms
    :param float goal_priority: the style priority for goal values
    """
    # First, we try to find groups of data to plot together in the same
    # color/style. We distinguish progress objects from statistical runs.
    goals: Final[Styler] = Styler(get_goal, goal_to_str, goal_priority,
                                  sort_by_name=False)
    algorithms: Final[Styler] = Styler(get_algorithm, "all algos",
                                       algo_priority)
    f_dim: Optional[str] = None
    t_dim: Optional[str] = None
    source: List[Ecdf] = cast(List[Ecdf], ecdfs) \
        if isinstance(ecdfs, list) else list(ecdfs)
    del ecdfs

    x_labels: Set[str] = set()

    # First pass: find out the goals and algorithms
    for ee in source:
        if not isinstance(ee, Ecdf):
            raise TypeError(f"Type {type(ee)} is not supported.")
        goals.add(ee)
        algorithms.add(ee)
        x_labels.add(ee.time_label())

        # Validate that we have consistent time and objective units.
        if f_dim is None:
            f_dim = ee.f_name
        elif f_dim != ee.f_name:
            raise ValueError(
                f"F-units {f_dim} and {ee.f_name} do not fit!")

        if t_dim is None:
            t_dim = ee.time_unit
        elif t_dim != ee.time_unit:
            raise ValueError(
                f"Time units {t_dim} and {ee.time_unit} do not fit!")

    if (f_dim is None) or (t_dim is None) or (len(source) <= 0):
        raise ValueError("Illegal state?")

    source.sort(key=sort_key)

    def __set_importance(st: Styler):
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
    groups: List[Styler] = list()
    goals.compile()
    algorithms.compile()

    if goals.count > 1:
        groups.append(goals)
    if algorithms.count > 1:
        groups.append(algorithms)

    if len(groups) > 0:
        groups.sort()
        groups[0].set_line_color(distinct_colors_func)

        if len(groups) > 1:
            groups[1].set_line_dash(distinct_line_dashes_func)

    # If we only have <= 2 groups, we can mark None and not-None values with
    # different importance.
    if goals.has_none and (goals.count > 1):
        __set_importance(goals)
    elif algorithms.has_none and (algorithms.count > 1):
        __set_importance(algorithms)

    # we will collect all lines to plot in plot_list
    plot_list: List[Dict] = list()

    # set up the axis rangers
    if callable(x_axis):
        x_axis = x_axis(t_dim)
    if not isinstance(x_axis, AxisRanger):
        raise TypeError(f"x_axis must be AxisRanger, but is {type(x_axis)}.")

    if callable(y_axis):
        y_axis = y_axis("ecdf")
    if not isinstance(y_axis, AxisRanger):
        raise TypeError(f"y_axis must be AxisRanger, but is {type(y_axis)}.")

    # first we collect all progress object
    for ee in source:
        style = pd.create_line_style()
        for g in groups:
            g.add_line_style(ee, style)
        x = ee.ecdf[:, 0]
        style["x"] = x
        x_axis.register_array(x)
        y = ee.ecdf[:, 1]
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
    if xgrid or ygrid:
        grid_lwd = importance_to_line_width_func(-1)
        if xgrid:
            axes.grid(axis="x", color=pd.GRID_COLOR, linewidth=grid_lwd)
        if ygrid:
            axes.grid(axis="y", color=pd.GRID_COLOR, linewidth=grid_lwd)

    max_x: float = x_axis.get_pinf_replacement()
    min_x: Optional[float] = x_axis.get_0_replacement() \
        if x_axis.log_scale else None

    # plot the lines
    for line in plot_list:
        x = line["x"]
        changed = False
        if np.isposinf(x[-1]):
            x = x.copy()
            x[-1] = max_x
            changed = True
        if (x[0] <= 0) and (min_x is not None):
            if not changed:
                changed = True
                x = x.copy()
            x[0] = min_x
        if changed:
            line["x"] = x
        axes.step(where="post", **line)
    del plot_list

    x_axis.apply(axes, "x")
    y_axis.apply(axes, "y")

    if legend:
        handles: List[Artist] = list()

        for g in groups:
            g.add_to_legend(handles)
            g.has_style = False

        if algorithms.has_style:
            algorithms.add_to_legend(handles)
        if goals.has_style:
            goals.add_to_legend(handles)

        axes.legend(loc="upper left",
                    handles=handles,
                    labelcolor=[art.color if hasattr(art, "color")
                                else pd.COLOR_BLACK for art in handles],
                    fontsize=font_size_0)

    pu.label_axes(axes=axes,
                  xlabel=xlabel(list(x_labels))
                  if callable(xlabel) else xlabel,
                  xlabel_inside=xlabel_inside,
                  xlabel_location=1,
                  ylabel=ylabel(f_dim) if callable(ylabel) else ylabel,
                  ylabel_inside=ylabel_inside,
                  ylabel_location=0,
                  font_size=font_size_0)
