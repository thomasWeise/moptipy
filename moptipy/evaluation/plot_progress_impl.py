"""Plot a set of progress objects into one figure."""
from typing import List, Dict, Tuple, Final, Callable, Iterable, Union, \
    Optional

from matplotlib.figure import Figure, SubplotBase  # type: ignore

from moptipy.evaluation.axis_ranger import AxisRanger
from moptipy.evaluation.plot_defaults import key_func_inst, \
    default_name_func, distinct_colors
from moptipy.evaluation.progress import Progress


def plot_progress(progresses: Iterable[Progress],
                  figure: Union[SubplotBase, Figure],
                  x_axis: Union[AxisRanger, Callable] = AxisRanger.for_axis,
                  y_axis: Union[AxisRanger, Callable] = AxisRanger.for_axis,
                  key_func: Callable = key_func_inst,
                  name_func: Callable = default_name_func,
                  distinct_colors_func: Callable = distinct_colors,
                  legend: bool = True) -> None:
    """
    Plot a set of progress lines into one chart.

    :param Iterable[moptipy.evaluation.Progress] progresses: the iterable
        of progresses
    :param Union[SubplotBase, Figure] figure: the figure to plot in
    :param Union[moptipy.evaluation.AxisRanger, Callable] x_axis: the x_axis
    :param Union[moptipy.evaluation.AxisRanger, Callable] y_axis: the y_axis
    :param Callable key_func: the function extracting the key from a progress
    :param Callable name_func: the function converting keys to names
    :param Callable distinct_colors_func: the function returning the palette
    :param bool legend: should we plot the legend?
    """
    # find groups of runs to plot together in the same color/style
    groups: Dict[object, List[Progress]] = dict()
    x_dim: Optional[str] = None
    y_dim: Optional[str] = None

    for prg in progresses:
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

        key = key_func(prg)
        if key in groups:
            groups[key].append(prg)
        else:
            groups[key] = [prg]

    n_groups: Final[int] = len(groups)
    if n_groups <= 0:
        raise ValueError("There must be at least one group.")
    if (x_dim is None) or (y_dim is None):
        raise ValueError("Illegal state?")

    # create consistent orderings
    for runs in groups.values():
        runs.sort()

    # now name the data
    key_list: List[object] = list(groups.keys())
    names_and_keys: Final[List[Tuple[str, object]]] = \
        [(name_func(key), key) for key in key_list]
    del key_list
    names_and_keys.sort()

    keys: Final[List[object]] = [key[1] for key in names_and_keys]
    names: Final[List[str]] = [key[0] for key in names_and_keys]
    del names_and_keys

    # set up the graphics area
    axes: Final = figure.add_axes([0.05, 0.05, 0.9, 0.9]) \
        if isinstance(figure, Figure) else figure.axes

    colors = distinct_colors_func(n_groups)

    if callable(x_axis):
        x_axis = x_axis(x_dim)
    if not isinstance(x_axis, AxisRanger):
        raise TypeError(f"x_axis must be AxisRanger, but is {type(x_axis)}.")

    if callable(y_axis):
        y_axis = y_axis(y_dim)
    if not isinstance(y_axis, AxisRanger):
        raise TypeError(f"y_axis must be AxisRanger, but is {type(y_axis)}.")

    # plot the lines in a round robin fashio
    found: bool = True
    order: bool = True
    while found:
        found = False
        steps = range(n_groups) if order else range(n_groups, 0, -1)
        for index in steps:
            key = keys[index]
            runs = groups[key]
            if runs:
                found = True
                run = runs.pop()
                x_axis.register_array(run.time)
                y_axis.register_array(run.f)
                axes.step(run.time, run.f, color=colors[index],
                          where="post")

    x_axis.apply(axes, "x")
    y_axis.apply(axes, "y")

    if legend:
        leg = axes.legend(loc="upper right",
                          labels=names,
                          labelcolor=colors)
        for i, lgd in enumerate(leg.legendHandles):
            lgd.set_color(colors[i])
