"""Default styles for plots."""
from typing import Union

import matplotlib.colors  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from moptipy.evaluation.base_classes import PerRunData, MultiRunData


def key_func_inst(data: Union[PerRunData, MultiRunData]) -> str:
    """
    Use the instance as a key for the dataset.

    :param data: the data to compute a key for
    :return: the corresponding key with only the instance name
    :rtype: str
    """
    if data.instance is None:
        raise ValueError("Instance cannot be None.")
    return data.instance


def key_func_algo(data: Union[PerRunData, MultiRunData]) -> str:
    """
    Use the algorithm as a key for the dataset.

    :param data: the data to compute a key for
    :return: the corresponding key with only the algorithm name
    :rtype: str
    """
    if data.algorithm is None:
        raise ValueError("Algorithm cannot be None.")
    return data.algorithm


def default_name_func(key) -> str:
    """
    Compute the default name for a `key`: `str(key)`.

    :param str key: the key
    :return: the key, too
    :rtype: str
    """
    return str(key)


def default_palette_func() -> matplotlib.colors.ListedColormap:
    """
    Get the default palette.

    :return: the default palette
    :rtype: matplotlib.colors.ListedColormap
    """
    return plt.get_cmap("tab10")
