"""Plot a set of progress objects into one figure."""
from typing import List, Dict, Tuple, Final, Callable, Iterable, Union, \
    Optional, Set, cast

import matplotlib.lines as mlines  # type: ignore
from matplotlib.artist import Artist  # type: ignore
from matplotlib.figure import Figure, SubplotBase  # type: ignore

import moptipy.evaluation.plot_defaults as pd
import moptipy.evaluation.plot_utils as pu
from moptipy.evaluation.axis_ranger import AxisRanger
from moptipy.evaluation.progress import Progress
from moptipy.evaluation.stat_run import StatRun


def plot_progress(progresses: Iterable[Union[Progress, StatRun]],
                  figure: Union[SubplotBase, Figure],
                  x_axis: Union[AxisRanger, Callable] = AxisRanger.for_axis,
                  y_axis: Union[AxisRanger, Callable] = AxisRanger.for_axis,
                  legend: bool = True,
                  key_func: Callable = pd.key_func_inst,
                  name_func: Callable = pd.default_name_func,
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
                  xlabel: Union[None, str, Callable] = pd.default_axis_label,
                  xlabel_inside: bool = True,
                  ylabel: Union[None, str, Callable] =
                  pd.default_axis_label,
                  ylabel_inside: bool = True) -> None:
    """
    Plot a set of progress or statistical run lines into one chart.

    :param Iterable[moptipy.evaluation.Progress] progresses: the iterable
        of progresses
    :param Union[SubplotBase, Figure] figure: the figure to plot in
    :param Union[moptipy.evaluation.AxisRanger, Callable] x_axis: the x_axis
    :param Union[moptipy.evaluation.AxisRanger, Callable] y_axis: the y_axis
    :param bool legend: should we plot the legend?
    :param Callable key_func: the function extracting the key from a progress
    :param Callable name_func: the function converting keys to names
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
    :param bool ylabel_inside: put the xyaxis label inside the plot (so that
        it does not consume additional horizontal space)
    """
    # First, we try to find groups of data to plot together in the same
    # color/style. We distinguish progress objects from statistical runs.
    groups: Dict[object, Tuple[List[Progress], List[StatRun]]] = dict()
    x_dim: Optional[str] = None
    y_dim: Optional[str] = None
    has_progress: bool = False
    has_statrun: bool = False
    stat_names_set: Set[str] = set()
    for prg in progresses:
        # Compute the key identifying the right group for prg.
        key = key_func(prg)

        gp: Tuple[List[Progress], List[StatRun]]
        if key in groups:
            gp = groups[key]
        else:
            groups[key] = gp = list(), list()

        # Check the type and decide to which list they should be added.
        if isinstance(prg, Progress):
            gp[0].append(prg)
            has_progress = True
        elif isinstance(prg, StatRun):
            has_statrun = True
            gp[1].append(prg)
            stat_names_set.add(prg.stat_name)
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

    # Check if there was useful data to plot.
    n_groups: Final[int] = len(groups)
    if n_groups <= 0:
        raise ValueError("There must be at least one group.")
    if (x_dim is None) or (y_dim is None) or \
            (not (has_progress or has_statrun)):
        raise ValueError("Illegal state?")

    # Create consistent orderings of the line groups.
    for runs in groups.values():
        for lst in runs:
            cast(List, lst).sort()

    # Now name the data consistently. For each key, one name
    # will be computed. The data elements are then sorted by
    # their names and the names can be used for the legend.
    key_list: List[object] = list(groups.keys())
    names_and_keys: Final[List[Tuple[str, object]]] = \
        [(name_func(key), key) for key in key_list]
    del key_list
    names_and_keys.sort()

    keys: Final[List[object]] = [key[1] for key in names_and_keys]
    names: Final[List[str]] = [key[0] for key in names_and_keys]
    del names_and_keys

    # Compute the importance values for the line types and get the
    # base style features line width and alpha.
    progress_importance = 0
    statrun_importance = 0
    if has_statrun and has_progress:
        progress_importance -= 1
        statrun_importance += 1

    progress_alpha = importance_to_alpha_func(progress_importance)
    statrun_alpha = importance_to_alpha_func(statrun_importance)
    progress_linewidth = importance_to_line_width_func(progress_importance)
    statrun_linewidth = importance_to_line_width_func(statrun_importance)
    colors = list(distinct_colors_func(n_groups))

    plot_list: List[Dict] = list()
    for groupidx, key in enumerate(keys):
        for prgs in groups[key][0]:
            plot_list.append(pd.create_line_style(
                x=prgs.time,
                y=prgs.f,
                color=colors[groupidx],
                alpha=progress_alpha,
                linewidth=progress_linewidth))

    # Perform some mild, deterministic shuffling to obtain a fair printing
    # order of lines: No line group should completely cover another one.
    lll = len(plot_list)
    if lll > 4:
        center = lll // 2
        for i in range(1, center, 2):
            plot_list[i], plot_list[-i] = plot_list[-i], plot_list[i]
        for start, end in [(center, lll - center), (lll - center, lll - 1)]:
            for i in range(lll // 4):
                plot_list[start + i], plot_list[end - i] = \
                    plot_list[end - i], plot_list[start + i]

    stat_names: Optional[List[str]] = None
    stat_dashes: Optional[Tuple] = None
    if has_statrun:
        # Obtain the names of the statistics, if any.
        stat_names = list(stat_names_set)
        stat_names.sort()
        stat_dashes = distinct_line_dashes_func(len(stat_names))

        for nameidx, stat_name in enumerate(stat_names):
            for groupidx, key in enumerate(keys):
                for strn in groups[key][1]:
                    if strn.stat_name == stat_name:
                        plot_list.append(pd.create_line_style(
                            x=strn.stat[:, 0],
                            y=strn.stat[:, 1],
                            color=colors[groupidx],
                            alpha=statrun_alpha,
                            linewidth=statrun_linewidth,
                            linestyle=stat_dashes[nameidx]))

    del stat_names_set

    font_size_0: Final[float] = importance_to_font_size_func(0)

    # set up the graphics area
    axes: Final = figure.add_axes([0.01, 0.01, 0.98, 0.98]) \
        if isinstance(figure, Figure) else figure.axes
    axes.tick_params(axis="x", labelsize=font_size_0)
    axes.tick_params(axis="y", labelsize=font_size_0)

    if xgrid or ygrid:
        grid_lwd = importance_to_line_width_func(-1)
        if xgrid:
            axes.grid(axis="x", color=pd.GRID_COLOR, linewidth=grid_lwd)
        if ygrid:
            axes.grid(axis="y", color=pd.GRID_COLOR, linewidth=grid_lwd)

    if callable(x_axis):
        x_axis = x_axis(x_dim)
    if not isinstance(x_axis, AxisRanger):
        raise TypeError(f"x_axis must be AxisRanger, but is {type(x_axis)}.")

    if callable(y_axis):
        y_axis = y_axis(y_dim)
    if not isinstance(y_axis, AxisRanger):
        raise TypeError(f"y_axis must be AxisRanger, but is {type(y_axis)}.")

    # plot the lines in a round robin fashion
    for line in plot_list:
        axes.step(where="post", **line)
        x_axis.register_array(line["x"])
        y_axis.register_array(line["y"])

    x_axis.apply(axes, "x")
    y_axis.apply(axes, "y")

    if legend:
        handles: List[Artist] = [mlines.Line2D([], [],
                                               color=colors[i],
                                               label=name)
                                 for i, name in enumerate(names)]

        args: Dict[str, object] = dict()

        if (len(stat_names) == 1) and (not has_progress):
            args["title"] = stat_names[0]
            args["title_fontsize"] = importance_to_font_size_func(1)
        else:
            handles.extend([mlines.Line2D([], [],
                                          color=pd.COLOR_BLACK,
                                          linewidth=statrun_linewidth,
                                          linestyle=stat_dashes[i],
                                          label=name)
                            for i, name in enumerate(stat_names)])
            colors.extend([pd.COLOR_BLACK] * len(stat_names))

        args["loc"] = "upper right"
        args["handles"] = handles
        args["labelcolor"] = colors
        args["fontsize"] = font_size_0

        axes.legend(**args)

    # put the label on the x-axis, if any
    if xlabel is not None:
        if callable(xlabel):
            xlabel = xlabel(x_dim)
        if not isinstance(xlabel, str):
            raise TypeError(f"xlabel must be str but is {type(xlabel)}.")
        if len(xlabel) > 0:
            if xlabel_inside:
                pu.label_box(axes,
                             text=xlabel,
                             x=0.5,
                             y=0,
                             font_size=font_size_0)
            else:
                axes.set_xlabel(xlabel,
                                fontsize=font_size_0)

    # put the label on the y-axis, if any
    if ylabel is not None:
        if callable(ylabel):
            ylabel = ylabel(y_dim)
        if not isinstance(ylabel, str):
            raise TypeError(f"ylabel must be str but is {type(ylabel)}.")
        if len(ylabel) > 0:
            if ylabel_inside:
                pu.label_box(axes,
                             text=ylabel,
                             x=0,
                             y=1,
                             font_size=font_size_0,
                             may_rotate_text=True)
            else:
                axes.set_ylabel(ylabel,
                                fontsize=font_size_0)
