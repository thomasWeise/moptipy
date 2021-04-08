"""Default styles for plots."""
from typing import Union, Tuple, cast, List

import matplotlib.cm as mplcm  # type: ignore
import matplotlib.colors as colors  # type: ignore
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


#: A set of predefined unique looking colors
__FIXED_COLORS: Tuple[Tuple[Tuple[float, float, float], ...], ...] = \
    (((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0),
      (0.6392156862745098, 0.14901960784313725, 0.14901960784313725),
      (1.0, 0.6470588235294118, 0.0), (0.0, 0.5333333333333333, 0.2),
      (0.4666666666666667, 0.5333333333333333, 1.0),
      (0.8117647058823529, 0.0, 1.0), (0.4666666666666667, 0.0, 0.4),
      (0.8, 0.8, 0.0),
      (0.6274509803921569, 0.6627450980392157, 0.6274509803921569)),
     ((0.9019607843137255, 0.09803921568627451, 0.29411764705882354),
      (0.23529411764705882, 0.7058823529411765, 0.29411764705882354),
      (1.0, 0.8823529411764706, 0.09803921568627451),
      (0.0, 0.5098039215686274, 0.7843137254901961),
      (0.9607843137254902, 0.5098039215686274, 0.19215686274509805),
      (0.5686274509803921, 0.11764705882352941, 0.7058823529411765),
      (0.27450980392156865, 0.9411764705882353, 0.9411764705882353),
      (0.9411764705882353, 0.19607843137254902, 0.9019607843137255),
      (0.8235294117647058, 0.9607843137254902, 0.23529411764705882),
      (0.9803921568627451, 0.7450980392156863, 0.7450980392156863),
      (0.0, 0.5019607843137255, 0.5019607843137255),
      (0.9019607843137255, 0.7450980392156863, 1.0),
      (0.6666666666666666, 0.43137254901960786, 0.1568627450980392),
      (0.8117647058823529, 0.792156862745098, 0.03137254901960784),
      (0.5019607843137255, 0.0, 0.0),
      (0.6666666666666666, 1.0, 0.7647058823529411),
      (0.5019607843137255, 0.5019607843137255, 0.0),
      (1.0, 0.8470588235294118, 0.6941176470588235),
      (0.0, 0.0, 0.5019607843137255),
      (0.5176470588235295, 0.5176470588235295, 0.4627450980392157)),
     ((0.9019607843137255, 0.09803921568627451, 0.29411764705882354),
      (0.23529411764705882, 0.7058823529411765, 0.29411764705882354),
      (1.0, 0.8823529411764706, 0.09803921568627451),
      (0.0, 0.5098039215686274, 0.7843137254901961),
      (0.9607843137254902, 0.5098039215686274, 0.19215686274509805),
      (0.5686274509803921, 0.11764705882352941, 0.7058823529411765),
      (0.27450980392156865, 0.9411764705882353, 0.9411764705882353),
      (0.9411764705882353, 0.19607843137254902, 0.9019607843137255),
      (0.8235294117647058, 0.9607843137254902, 0.23529411764705882),
      (0.9803921568627451, 0.7450980392156863, 0.7450980392156863),
      (0.0, 0.5019607843137255, 0.5019607843137255),
      (0.9019607843137255, 0.7450980392156863, 1.0),
      (0.6666666666666666, 0.43137254901960786, 0.1568627450980392),
      (0.8117647058823529, 0.792156862745098, 0.03137254901960784),
      (0.5019607843137255, 0.0, 0.0),
      (0.6666666666666666, 1.0, 0.7647058823529411),
      (0.5019607843137255, 0.5019607843137255, 0.0),
      (1.0, 0.8470588235294118, 0.6941176470588235),
      (0.0, 0.0, 0.5019607843137255),
      (0.3137254901960784, 0.3137254901960784, 0.0),
      (0.5647058823529412, 0.6274509803921569, 0.6274509803921569)))


def distinct_colors(n: int) -> Tuple[Tuple[float, float, float], ...]:
    """
    Obtain a set of `n` distinct colors.

    :param int n: the number of colors required
    :return: a tuple of colors
    :rtype: Tuple[Tuple[float, float, float], ...]
    """
    if not isinstance(n, int):
        raise TypeError(f"n must be int but is {type(n)}.")
    if not (0 < n < 1000):
        raise ValueError(f"Invalid n={n}.")

    for k in __FIXED_COLORS:
        lk = len(k)
        if lk >= n:
            if lk == n:
                return k
            return tuple(k[0:n])

    cm = plt.get_cmap('gist_rainbow')
    cNorm = colors.Normalize(vmin=0, vmax=n - 1)
    scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
    qq = cast(List[Tuple[float, float, float]],
              [tuple(scalarMap.to_rgba(i)[0:3]) for i in range(n)])
    ss = set(qq)
    if len(ss) != n:
        raise ValueError(f"Could not obtain {n} distinct colors.")
    return tuple(qq)
