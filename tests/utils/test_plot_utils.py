"""Test the plot utilities."""

from typing import List

from matplotlib.figure import Figure

import moptipy.utils.plot_utils as pu
from moptipy.utils.temp import TempDir


def test_create_and_save_figure():
    """Test figure creation and saving."""
    f = pu.create_figure(10, 10)
    assert isinstance(f, Figure)
    ax = pu.get_axes(f)
    # assert isinstance(f, Axes)
    ax.hist(x=range(100))
    with TempDir.create() as td:
        res = pu.save_figure(fig=f, file_name="test", dir_name=td,
                             formats=["svg", "pdf"])
        assert isinstance(res, List)
        assert len(res) == 2
        assert td.resolve_inside("test.pdf") in res
        assert td.resolve_inside("test.svg") in res
        for k in res:
            k.enforce_file()
