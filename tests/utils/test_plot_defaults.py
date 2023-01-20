"""Test the plot defaults."""

from math import isfinite

from moptipy.utils.plot_defaults import (
    distinct_colors,
    distinct_markers,
    importance_to_alpha,
    importance_to_font_size,
)


def test_distinct_colors() -> None:
    """Test distinct colors."""
    for n in list({*list(range(1, 34)), 50, 64, 75, 96, 100,
                   128, 150, 200, 256}):
        colors = distinct_colors(n)
        assert isinstance(colors, tuple)
        assert len(colors) == n
        assert len(set(colors)) == n
        for col in colors:
            assert isinstance(col, tuple)
            assert len(col) == 3
            for c in col:
                assert isinstance(c, float)
                assert isfinite(c)
                assert 0 <= c <= 1


def test_distinct_markers() -> None:
    """Test the distinct markers."""
    for n in range(1, 8):
        markers = distinct_markers(n)
        assert isinstance(markers, tuple)
        assert len(markers) == n
        assert len(set(markers)) == n
        for s in markers:
            assert isinstance(s, str)
            assert len(s) == 1


def test_importance_to_alpha() -> None:
    """Test importance_to_alpha."""
    for i in range(-4, 10):
        imp = importance_to_alpha(i)
        assert isinstance(imp, float)
        assert isfinite(imp)
        assert 0 < imp <= 1
        assert (i <= 0) or (imp >= 1)


def test_importance_to_font_size() -> None:
    """Test importance_to_font_size."""
    for i in range(-4, 10):
        fs = importance_to_font_size(i)
        assert isinstance(fs, float)
        assert isfinite(fs)
        assert 6 < fs <= 20
