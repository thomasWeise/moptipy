"""
Plot a set of ECDF or ERT-ECDF objects into one figure.

The empirical cumulative distribution function (ECDF, see
:mod:`~moptipy.evaluation.ecdf`) is a function that shows the fraction of runs
that were successful in attaining a certain goal objective value over the
time. The combination of ERT and ECDF is discussed in
:mod:`~moptipy.evaluation.ertecdf`.

1. Nikolaus Hansen, Anne Auger, Steffen Finck, Raymond Ros. *Real-Parameter
   Black-Box Optimization Benchmarking 2010: Experimental Setup.*
   Research Report RR-7215, INRIA. 2010. inria-00462481.
   https://hal.inria.fr/inria-00462481/document/
2. Dave Andrew Douglas Tompkins and Holger H. Hoos. UBCSAT: An Implementation
   and Experimentation Environment for SLS Algorithms for SAT and MAX-SAT. In
   *Revised Selected Papers from the Seventh International Conference on
   Theory and Applications of Satisfiability Testing (SAT'04),* May 10-13,
   2004, Vancouver, BC, Canada, pages 306-320. Lecture Notes in Computer
   Science (LNCS), volume 3542. Berlin, Germany: Springer-Verlag GmbH.
   ISBN: 3-540-27829-X. doi: https://doi.org/10.1007/11527695_24.
3. Holger H. Hoos and Thomas Stützle. Evaluating Las Vegas Algorithms -
   Pitfalls and Remedies. In Gregory F. Cooper and Serafín Moral, editors,
   *Proceedings of the 14th Conference on Uncertainty in Artificial
   Intelligence (UAI'98)*, July 24-26, 1998, Madison, WI, USA, pages 238-245.
   San Francisco, CA, USA: Morgan Kaufmann Publishers Inc.
   ISBN: 1-55860-555-X.
"""
from math import inf, isfinite
from typing import Any, Callable, Final, Iterable, cast

import numpy as np
from matplotlib.artist import Artist  # type: ignore
from matplotlib.axes import Axes  # type: ignore
from matplotlib.figure import Figure  # type: ignore

import moptipy.utils.plot_defaults as pd
import moptipy.utils.plot_utils as pu
from moptipy.evaluation.axis_ranger import AxisRanger
from moptipy.evaluation.base import get_algorithm, sort_key
from moptipy.evaluation.ecdf import Ecdf, get_goal, goal_to_str
from moptipy.evaluation.styler import Styler
from moptipy.utils.lang import Lang
from moptipy.utils.types import type_error


def plot_ecdf(ecdfs: Iterable[Ecdf],
              figure: Axes | Figure,
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
              x_label: None | str | Callable[[str], str] =
              lambda x: x if isinstance(x, str) else x[0],
              x_label_inside: bool = True,
              y_label: None | str | Callable[[str], str] =
              Lang.translate_func("ECDF"),
              y_label_inside: bool = True,
              algorithm_priority: float = 5.0,
              goal_priority: float = 0.333,
              algorithm_sort_key: Callable[[str], Any] = lambda x: x,
              goal_sort_key: Callable[[str], Any] = lambda x: x,
              algorithm_namer: Callable[[str], str] = lambda x: x,
              color_algorithms_as_fallback_group: bool = True) -> Axes:
    """
    Plot a set of ECDF functions into one chart.

    :param ecdfs: the iterable of ECDF functions
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
    :param algorithm_priority: the style priority for algorithms
    :param goal_priority: the style priority for goal values
    :param algorithm_namer: the name function for algorithms receives an
        algorithm ID and returns an instance name; default=identity function
    :param color_algorithms_as_fallback_group: if only a single group of data
        was found, use algorithms as group and put them in the legend
    :param algorithm_sort_key: the sort key function for algorithms
    :param goal_sort_key: the sort key function for goals
    :returns: the axes object to allow you to add further plot elements
    """
    # Before doing anything, let's do some type checking on the parameters.
    # I want to ensure that this function is called correctly before we begin
    # to actually process the data. It is better to fail early than to deliver
    # some incorrect results.
    if not isinstance(ecdfs, Iterable):
        raise type_error(ecdfs, "ecdfs", Iterable)
    if not isinstance(figure, Axes | Figure):
        raise type_error(figure, "figure", (Axes, Figure))
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
    if not isinstance(algorithm_priority, float):
        raise type_error(algorithm_priority, "algorithm_priority", float)
    if not isfinite(algorithm_priority):
        raise ValueError(f"algorithm_priority cannot be {algorithm_priority}.")
    if not isfinite(goal_priority):
        raise ValueError(f"goal_priority cannot be {goal_priority}.")
    if not callable(algorithm_namer):
        raise type_error(algorithm_namer, "algorithm_namer", call=True)
    if not callable(algorithm_sort_key):
        raise type_error(algorithm_sort_key, "algorithm_sort_key", call=True)
    if not callable(goal_sort_key):
        raise type_error(goal_sort_key, "goal_sort_key", call=True)

    # First, we try to find groups of data to plot together in the same
    # color/style. We distinguish progress objects from statistical runs.
    goals: Final[Styler] = Styler(key_func=get_goal,
                                  namer=goal_to_str,
                                  priority=goal_priority,
                                  name_sort_function=goal_sort_key)
    algorithms: Final[Styler] = Styler(key_func=get_algorithm,
                                       namer=algorithm_namer,
                                       none_name=Lang.translate("all_algos"),
                                       priority=algorithm_priority,
                                       name_sort_function=algorithm_sort_key)
    f_dim: str | None = None
    t_dim: str | None = None
    source: list[Ecdf] = cast(list[Ecdf], ecdfs) \
        if isinstance(ecdfs, list) else list(ecdfs)
    del ecdfs

    x_labels: set[str] = set()

    # First pass: find out the goals and algorithms
    for ee in source:
        if not isinstance(ee, Ecdf):
            raise type_error(ee, "data source", Ecdf)
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

    if f_dim is None:
        raise ValueError("f_dim cannot be None")
    if t_dim is None:
        raise ValueError("t_dim cannot be None")
    if (source is None) or (len(source) <= 0):
        raise ValueError(f"source cannot be {source}.")

    # determine the style groups
    groups: list[Styler] = []
    goals.finalize()
    algorithms.finalize()

    # pick the right sorting order
    sf: Callable[[Ecdf], Any] = sort_key
    if (goals.count > 1) and (algorithms.count == 1):
        def __x1(r: Ecdf, ssf=goal_sort_key) -> Any:
            return ssf(goal_to_str(r.goal_f))
        sf = __x1
    elif (goals.count == 1) and (algorithms.count > 1):
        def __x2(r: Ecdf, ssf=algorithm_sort_key) -> Any:
            return ssf(r.algorithm)
        sf = __x2
    elif (goals.count > 1) and (algorithms.count > 1):
        def __x3(r: Ecdf, sgs=goal_sort_key, sas=algorithm_sort_key,
                 ag=algorithm_priority > goal_priority) -> tuple[Any, Any]:
            k1 = sgs(goal_to_str(r.goal_f))
            k2 = sas(r.algorithm)
            return (k2, k1) if ag else (k1, k2)
        sf = __x3

    source.sort(key=sf)

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

    if goals.count > 1:
        groups.append(goals)
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
    if goals.has_none and (goals.count > 1):
        __set_importance(goals)
    elif algorithms.has_none and (algorithms.count > 1):
        __set_importance(algorithms)

    # we will collect all lines to plot in plot_list
    plot_list: list[dict] = []

    # set up the axis rangers
    if callable(x_axis):
        x_axis = x_axis(t_dim)
    if not isinstance(x_axis, AxisRanger):
        raise type_error(x_axis, "x_axis", AxisRanger)

    if callable(y_axis):
        y_axis = y_axis("ecdf")
    if not isinstance(y_axis, AxisRanger):
        raise type_error(y_axis, "y_axis", AxisRanger)

    # first we collect all ecdf object
    max_time: int | float = -inf
    max_ecdf: int | float = -inf
    max_ecdf_is_at_max_time: bool = False
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

        # We need to detect the special case that the maximum time is at
        # the maximum ECDF value. In this case, we will later need to extend
        # the visible area of the x-axis.
        if len(x) < 2:
            continue
        fy = y[-2]
        ft = x[-2]
        if isfinite(ft):
            if fy >= max_ecdf:
                if fy > max_ecdf:
                    max_ecdf_is_at_max_time = (ft >= max_time)
                    max_ecdf = fy
                else:
                    max_ecdf_is_at_max_time = max_ecdf_is_at_max_time \
                        or (ft >= max_time)
            elif ft > max_time:
                max_ecdf_is_at_max_time = False
            if ft > max_time:
                max_time = ft
    del source

    font_size_0: Final[float] = importance_to_font_size_func(0)

    # If the maximum of any ECDF is located directly at the end of the
    # x-axis, we need to slightly extend the axis to make it visible.
    if max_ecdf_is_at_max_time:
        x_axis.pad_detected_range(pad_max=True)

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

    max_x: float = x_axis.get_pinf_replacement()
    min_x: float | None = x_axis.get_0_replacement() \
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
        handles: list[Artist] = []

        for g in groups:
            g.add_to_legend(handles.append)
            g.has_style = False

        if algorithms.has_style:
            algorithms.add_to_legend(handles.append)
        if goals.has_style:
            goals.add_to_legend(handles.append)

        axes.legend(loc="upper left",
                    handles=handles,
                    labelcolor=[art.color if hasattr(art, "color")
                                else pd.COLOR_BLACK for art in handles],
                    fontsize=font_size_0)

    pu.label_axes(axes=axes,
                  x_label=" ".join([x_label(x) for x in sorted(x_labels)])
                  if callable(x_label) else x_label,
                  x_label_inside=x_label_inside,
                  x_label_location=1,
                  y_label=y_label(f_dim) if callable(y_label) else y_label,
                  y_label_inside=y_label_inside,
                  y_label_location=0,
                  font_size=font_size_0)
    return axes
