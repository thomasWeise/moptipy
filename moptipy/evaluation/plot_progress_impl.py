"""Plot a set of `Progress` or `StatRun` objects into one figure."""
from typing import List, Dict, Final, Callable, Iterable, Union, \
    Optional

from matplotlib.artist import Artist  # type: ignore
from matplotlib.axes import Axes  # type: ignore
from matplotlib.figure import Figure, SubplotBase  # type: ignore

import moptipy.evaluation.plot_defaults as pd
import moptipy.evaluation.plot_utils as pu
from moptipy.evaluation.axis_ranger import AxisRanger
from moptipy.evaluation.base import get_instance, get_algorithm, sort_key
from moptipy.evaluation.lang import Lang
from moptipy.evaluation.progress import Progress
from moptipy.evaluation.stat_run import StatRun, get_statistic
from moptipy.evaluation.styler import Styler


def plot_progress(progresses: Iterable[Union[Progress, StatRun]],
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
                  xlabel: Union[None, str, Callable] = Lang.translate,
                  xlabel_inside: bool = True,
                  ylabel: Union[None, str, Callable] = Lang.translate,
                  ylabel_inside: bool = True,
                  inst_priority: float = 0.666,
                  algo_priority: float = 0.333,
                  stat_priority: float = 0.0) -> None:
    """
    Plot a set of progress or statistical run lines into one chart.

    :param progresses: the iterable of progresses and statistical runs
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
    :param float inst_priority: the style priority for instances
    :param float algo_priority: the style priority for algorithms
    :param float stat_priority: the style priority for statistics
    """
    # First, we try to find groups of data to plot together in the same
    # color/style. We distinguish progress objects from statistical runs.
    instances: Final[Styler] = Styler(get_instance, "all insts",
                                      inst_priority)
    algorithms: Final[Styler] = Styler(get_algorithm, "all algos",
                                       algo_priority)
    statistics: Final[Styler] = Styler(get_statistic, "single run",
                                       stat_priority)
    x_dim: Optional[str] = None
    y_dim: Optional[str] = None
    progress_list: List[Progress] = []
    statrun_list: List[StatRun] = []

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
            raise TypeError("Invalid progress object: "
                            f"type {type(prg)} is not supported.")

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

    if (x_dim is None) or (y_dim is None) or \
            ((len(progress_list) + len(statrun_list)) <= 0):
        raise ValueError("Illegal state?")

    statrun_list.sort(key=sort_key)
    progress_list.sort()
    instances.compile()
    algorithms.compile()
    statistics.compile()

    def __set_importance(st: Styler):
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
    groups: List[Styler] = []

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
    plot_list: List[Dict] = []

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
    if xgrid or ygrid:
        grid_lwd = importance_to_line_width_func(-1)
        if xgrid:
            axes.grid(axis="x", color=pd.GRID_COLOR, linewidth=grid_lwd)
        if ygrid:
            axes.grid(axis="y", color=pd.GRID_COLOR, linewidth=grid_lwd)

    # set up the axis rangers
    if callable(x_axis):
        x_axis = x_axis(x_dim)
    if not isinstance(x_axis, AxisRanger):
        raise TypeError(f"x_axis must be AxisRanger, but is {type(x_axis)}.")

    if callable(y_axis):
        y_axis = y_axis(y_dim)
    if not isinstance(y_axis, AxisRanger):
        raise TypeError(f"y_axis must be AxisRanger, but is {type(y_axis)}.")

    # plot the lines
    for line in plot_list:
        axes.step(where="post", **line)
        x_axis.register_array(line["x"])
        y_axis.register_array(line["y"])
    del plot_list

    x_axis.apply(axes, "x")
    y_axis.apply(axes, "y")

    if legend:
        handles: List[Artist] = []

        for g in groups:
            g.add_to_legend(handles)
            g.has_style = False

        if instances.has_style:
            instances.add_to_legend(handles)
        if algorithms.has_style:
            algorithms.add_to_legend(handles)
        if statistics.has_style:
            statistics.add_to_legend(handles)

        axes.legend(loc="upper right",
                    handles=handles,
                    labelcolor=[art.color if hasattr(art, "color")
                                else pd.COLOR_BLACK for art in handles],
                    fontsize=font_size_0)

    pu.label_axes(axes=axes,
                  xlabel=xlabel(x_dim) if callable(xlabel) else xlabel,
                  xlabel_inside=xlabel_inside,
                  xlabel_location=0.5,
                  ylabel=ylabel(y_dim) if callable(ylabel) else ylabel,
                  ylabel_inside=ylabel_inside,
                  ylabel_location=1,
                  font_size=font_size_0)
