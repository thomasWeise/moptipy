"""Test the unconstrained real vector space."""
from math import inf

import numpy as np

from moptipy.api.space import Space
from moptipy.spaces.vectorspace import VectorSpace
from moptipy.tests.space import validate_space
from moptipy.utils.logger import FileLogger
from moptipy.utils.temp import TempFile


def test_vector_space() -> None:
    """Test the vector space."""
    f = VectorSpace(12)
    assert isinstance(f, Space)
    assert str(f) == "r12d"

    a = f.create()
    assert isinstance(a, np.ndarray)
    assert len(a) == 12
    assert a.dtype == f.dtype

    for i in range(len(a)):
        a[i] = 1.0 / (i + 1.0)
    b = f.create()
    assert not f.is_equal(a, b)

    f.copy(a, b)
    assert f.is_equal(a, b)

    b[0] = 0.001
    assert not f.is_equal(a, b)

    with TempFile.create() as path:
        with FileLogger(path) as log:
            with log.key_values("F") as kv:
                f.log_parameters_to(kv)
        result = path.read_all_list()
    assert result == ["BEGIN_F",
                      "name: r12d",
                      "class: moptipy.spaces.vectorspace.VectorSpace",
                      "nvars: 12",
                      "dtype: d",
                      "lb: 0",
                      "ub: 1",
                      "END_F"]

    text = f.to_str(b)
    assert isinstance(text, str)
    assert len(text) > 0

    a = f.from_str(text)
    assert f.is_equal(a, b)

    def _invalid(x) -> np.ndarray:
        x[0] = inf
        return x

    def _valid(x) -> np.ndarray:
        nonlocal f
        return np.clip(x, f.lower_bound, f.upper_bound, x)
    validate_space(f, make_element_invalid=_invalid,
                   make_element_valid=_valid)


def test_bounded_vector_space_1() -> None:
    """Test the bounded vector space."""
    f = VectorSpace(3,
                    lower_bound=(1.0, 2.0, 3.0),
                    upper_bound=4.0)
    assert isinstance(f, Space)
    assert str(f) == "r3d"

    a = f.create()
    assert isinstance(a, np.ndarray)
    assert len(a) == 3
    assert a.dtype == f.dtype

    for i in range(len(a)):
        a[i] = f.lower_bound[i] + 0.1
    b = f.create()
    assert not f.is_equal(a, b)

    f.copy(b, a)
    assert f.is_equal(a, b)

    b[0] = 2
    assert not f.is_equal(a, b)

    with TempFile.create() as path:
        with FileLogger(path) as log:
            with log.key_values("F") as kv:
                f.log_parameters_to(kv)
        result = path.read_all_list()
    assert result == ["BEGIN_F",
                      "name: r3d",
                      "class: moptipy.spaces.vectorspace.VectorSpace",
                      "nvars: 3",
                      "dtype: d",
                      "lb: 1;2;3",
                      "ub: 4",
                      "END_F"]

    text = f.to_str(b)
    assert isinstance(text, str)
    assert len(text) > 0

    a = f.from_str(text)
    assert f.is_equal(a, b)

    def _invalid(x) -> np.ndarray:
        x[0] = inf
        return x

    def _valid(x) -> np.ndarray:
        nonlocal f
        return np.clip(x, f.lower_bound, f.upper_bound, x)

    validate_space(f, make_element_invalid=_invalid,
                   make_element_valid=_valid)


def test_bounded_vector_space_2() -> None:
    """Test the bounded vector space."""
    f = VectorSpace(3,
                    lower_bound=-1.0,
                    upper_bound=[4.0, 2.3, 7])
    assert isinstance(f, Space)
    assert str(f) == "r3d"

    a = f.create()
    assert isinstance(a, np.ndarray)
    assert len(a) == 3
    assert a.dtype == f.dtype

    for i in range(len(a)):
        a[i] = f.lower_bound[i] + 0.1
    b = f.create()
    assert not f.is_equal(a, b)

    f.copy(b, a)
    assert f.is_equal(a, b)

    b[0] = 2
    assert not f.is_equal(a, b)

    with TempFile.create() as path:
        with FileLogger(path) as log:
            with log.key_values("F") as kv:
                f.log_parameters_to(kv)
        result = path.read_all_list()
    assert result == ["BEGIN_F",
                      "name: r3d",
                      "class: moptipy.spaces.vectorspace.VectorSpace",
                      "nvars: 3",
                      "dtype: d",
                      "lb: -1",
                      "ub: 4;2.3;7",
                      "END_F"]

    text = f.to_str(b)
    assert isinstance(text, str)
    assert len(text) > 0

    a = f.from_str(text)
    assert f.is_equal(a, b)

    def _invalid(x) -> np.ndarray:
        x[0] = inf
        return x

    def _valid(x) -> np.ndarray:
        nonlocal f
        return np.clip(x, f.lower_bound, f.upper_bound, x)

    validate_space(f, make_element_invalid=_invalid,
                   make_element_valid=_valid)
