"""Test the plot utilities."""

from matplotlib.axes import Axes  # type: ignore
from matplotlib.figure import Figure  # type: ignore

import moptipy.utils.plot_utils as pu
from moptipy.utils.temp import TempDir


def test_create_and_save_figure() -> None:
    """Test figure creation and saving."""
    f = pu.create_figure(10, 10)
    assert isinstance(f, Figure)
    ax = pu.get_axes(f)
    ax.hist(x=range(100))
    with TempDir.create() as td:
        res = pu.save_figure(fig=f, file_name="test", dir_name=td,
                             formats=["svg", "pdf"])
        assert isinstance(res, list)
        assert len(res) == 2
        assert td.resolve_inside("test.pdf") in res
        assert td.resolve_inside("test.svg") in res
        for k in res:
            k.enforce_file()


def test_create_multi_figure() -> None:
    """Test the creation of a data sequence figure."""
    res = pu.create_figure_with_subplots(items=100,
                                         max_items_per_plot=10,
                                         max_rows=30,
                                         max_cols=5)
    assert isinstance(res, tuple)
    assert isinstance(res[0], Figure)
    inner = res[1]
    assert isinstance(inner, tuple)
    xy: set[tuple[int, int]] = set()
    ids: set[int] = set()
    data: set[int] = set()
    for fig, first, last, x, y, idx in inner:
        assert isinstance(fig, Axes)
        assert isinstance(first, int)
        assert isinstance(last, int)
        for idsx in range(first, last):
            assert idsx not in data
            data.add(idsx)
        assert isinstance(x, int)
        assert isinstance(y, int)
        assert 0 <= x < 5
        assert 0 <= y < 30
        xyy = (x, y)
        assert xyy not in xy
        xy.add(xyy)
        assert isinstance(idx, int)
        assert idx not in ids
        assert 0 <= idx <= 150
        ids.add(idx)

    for k in range(0, 100):
        assert k in data
    for k in range(min(ids), max(ids) + 1):
        assert k in ids
