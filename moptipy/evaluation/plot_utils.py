"""Utilities for creating and storing figures."""
import os.path
from math import sqrt, isfinite
from typing import Final, Iterable, Union, List

import matplotlib.pyplot  # type: ignore
from matplotlib import figure  # type: ignore

from moptipy.utils.io import dir_ensure_exists, file_ensure_exists, \
    enforce_file

#: the golden ratio
__GOLDEN_RATIO: Final[float] = 0.5 + (0.5 * sqrt(5))


def create_figure(width: Union[float, int, None] = 8.6,
                  height: Union[float, int, None] = None,
                  dpi: Union[float, int, None] = 384.0,
                  **kwargs) -> figure.Figure:
    """
    Create a matplotlib figure.

    :param Union[float,int,None] width: the optional width
    :param Union[float,int,None] height: the optional height
    :param Union[float,int,None] dpi: the dpi value
    :param kwargs: a set of optional arguments
    :return: the figure option
    :rtype: figure.Figure
    """
    # Check the size, i.e., width and height and use
    put_size: bool = True
    if width is None:
        if height is None:
            put_size = False
        else:
            width = __GOLDEN_RATIO * height
    elif height is None:
        height = width / __GOLDEN_RATIO

    if put_size:
        if isinstance(height, int):
            height = float(height)
        if not isinstance(height, float):
            raise ValueError(f"Invalid height type {type(height)}.")
        if not (isfinite(height) and (0.1 < height < 10000.0)):
            raise ValueError(f"Invalid height {height}.")
        nheight = int(0.5 + (height * 72)) / 72.0
        if not (isfinite(nheight) and (0.1 < nheight < 10000.0)):
            raise ValueError(f"Invalid height {height} as it maps to "
                             f"{nheight} after to-point-and-back conversion.")

        if isinstance(width, int):
            width = float(width)
        if not isinstance(width, float):
            raise ValueError(f"Invalid width type {type(width)}.")
        if not (isfinite(width) and (0.1 < width < 10000.0)):
            raise ValueError(f"Invalid width {width}.")
        nwidth = int(0.5 + (width * 72)) / 72.0
        if not (isfinite(nwidth) and (0.1 < nwidth < 10000.0)):
            raise ValueError(f"Invalid width {width} as it maps to "
                             f"{nwidth} after to-point-and-back conversion.")

        kwargs['figsize'] = width, height

    if dpi is not None:
        if isinstance(dpi, int):
            dpi = float(dpi)
        if not isinstance(dpi, float):
            raise ValueError(f"Invalid dpi type {type(dpi)}.")
        if not (isfinite(dpi) and (1.0 < dpi < 10000.0)):
            raise ValueError(f"Invalid dpi value {dpi}.")
        kwargs['dpi'] = dpi

    if 'frameon' not in kwargs:
        kwargs['frameon'] = False

    if 'constrained_layout' not in kwargs:
        kwargs['constrained_layout'] = False

    return figure.Figure(**kwargs)


def save_figure(fig: figure.Figure,
                file_name: str = "figure",
                dir_name: str = ".",
                formats: Union[str, Iterable[str]] = "svg") -> List[str]:
    """
    Store the given figure in files of the given formats and dispose it.

    :param figure.Figure fig: the figure to save
    :param str file_name: the basic file name
    :param str dir_name: the directory name
    :param Union[str, Iterable[str]] formats: the file format(s)
    :return: a list of files
    :rtype: List[str]
    """
    if not isinstance(fig, figure.Figure):
        raise TypeError(f"Invalid figure type {type(fig)}.")
    if not isinstance(file_name, str):
        raise TypeError(f"Invalid file_name type {type(file_name)}.")
    if len(file_name) <= 0:
        raise ValueError(f"Invalid filename '{file_name}'.")
    if not isinstance(dir_name, str):
        raise TypeError(f"Invalid dirname type {type(dir_name)}.")
    if len(dir_name) <= 0:
        raise ValueError(f"Invalid dirname '{dir_name}'.")
    if isinstance(formats, str):
        formats = [formats]
    if not isinstance(formats, Iterable):
        raise TypeError(f"Invalid format type {type(formats)}.")

    size = fig.get_size_inches()
    if size[0] >= size[1]:
        orientation = "landscape"
    else:
        orientation = "portrait"

    # set minimal margins to the axes to avoid wasting space
    for ax in fig.axes:
        ax.margins(0, 0)

    dir_name = dir_ensure_exists(dir_name)
    files = list()
    for fmt in formats:
        if not isinstance(fmt, str):
            raise TypeError(f"Invalid format type {type(fmt)}.")
        dest_file, _ = file_ensure_exists(
            os.path.join(dir_name, f"{file_name}.{fmt}"))
        fig.savefig(dest_file, transparent=True, format=fmt,
                    orientation=orientation,
                    dpi="figure",
                    bbox_inches='tight',
                    pad_inches=1.0 / 72.0)
        files.append(enforce_file(dest_file))

    fig.clf(False)
    matplotlib.pyplot.close(fig)
    del fig

    if len(files) <= 0:
        raise ValueError("No formats were specified.")

    return files
