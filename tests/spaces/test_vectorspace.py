"""Test the unconstrained real vector space."""
from math import inf

import numpy as np

# noinspection PyPackageRequirements
from pytest import raises

from moptipy.api.space import Space
from moptipy.spaces.vectorspace import VectorSpace
from moptipy.tests.space import validate_space
from moptipy.utils.logger import FileLogger
from moptipy.utils.temp import TempFile


def test_vector_space_1() -> None:
    """Test the vector space."""
    space = VectorSpace(12)
    assert isinstance(space, Space)
    assert str(space) == "r12d"
    assert space.lower_bound_all_same
    assert space.upper_bound_all_same
    assert all(space.lower_bound == 0.0)
    assert all(space.upper_bound == 1.0)

    a = space.create()
    assert isinstance(a, np.ndarray)
    assert len(a) == 12
    assert a.dtype == space.dtype
    assert space.to_str(space.create()) == "0;0;0;0;0;0;0;0;0;0;0;0"

    for i in range(len(a)):
        a[i] = 1.0 / (i + 1.0)
    b = space.create()
    assert not space.is_equal(a, b)

    space.copy(a, b)
    assert space.is_equal(a, b)

    b[0] = 0.001
    assert not space.is_equal(a, b)

    with TempFile.create() as path:
        with FileLogger(path) as log, log.key_values("F") as kv:
            space.log_parameters_to(kv)
        result = path.read_all_list()
    assert result == ["BEGIN_F",
                      "name: r12d",
                      "class: moptipy.spaces.vectorspace.VectorSpace",
                      "nvars: 12",
                      "dtype: d",
                      "lb: 0",
                      "ub: 1",
                      "END_F"]

    text = space.to_str(b)
    assert isinstance(text, str)
    assert len(text) > 0

    a = space.from_str(text)
    assert space.is_equal(a, b)

    def _invalid(x) -> np.ndarray:
        x[0] = inf
        return x

    def _valid(x) -> np.ndarray:
        nonlocal space
        return np.clip(x, space.lower_bound, space.upper_bound, x)
    validate_space(space, make_element_invalid=_invalid,
                   make_element_valid=_valid)


def test_vector_space_2() -> None:
    """Test the bounded vector space."""
    space = VectorSpace(3,
                        lower_bound=(1.0, 2.0, 3.0),
                        upper_bound=4.0)
    assert isinstance(space, Space)
    assert str(space) == "r3d"
    assert not space.lower_bound_all_same
    assert space.upper_bound_all_same
    assert all(space.upper_bound == 4.0)
    assert all(space.lower_bound == np.array([1.0, 2.0, 3.0]))

    a = space.create()
    assert isinstance(a, np.ndarray)
    assert len(a) == 3
    assert a.dtype == space.dtype

    for i in range(len(a)):
        a[i] = space.lower_bound[i] + 0.1
    b = space.create()
    assert not space.is_equal(a, b)

    space.copy(b, a)
    assert space.is_equal(a, b)

    b[0] = 2
    assert not space.is_equal(a, b)

    with TempFile.create() as path:
        with FileLogger(path) as log, log.key_values("F") as kv:
            space.log_parameters_to(kv)
        result = path.read_all_list()
    assert result == ["BEGIN_F",
                      "name: r3d",
                      "class: moptipy.spaces.vectorspace.VectorSpace",
                      "nvars: 3",
                      "dtype: d",
                      "lb: 1;2;3",
                      "ub: 4",
                      "END_F"]

    text = space.to_str(b)
    assert text == "2;2.1;3.1"
    assert isinstance(text, str)
    assert len(text) > 0

    a = space.from_str(text)
    assert space.is_equal(a, b)

    def _invalid(x) -> np.ndarray:
        x[0] = inf
        return x

    def _valid(x) -> np.ndarray:
        nonlocal space
        return np.clip(x, space.lower_bound, space.upper_bound, x)

    validate_space(space, make_element_invalid=_invalid,
                   make_element_valid=_valid)


def test_vector_space_3() -> None:
    """Test the bounded vector space."""
    space = VectorSpace(3,
                        lower_bound=-1.0,
                        upper_bound=[4.0, 2.3, 7])
    assert isinstance(space, Space)
    assert str(space) == "r3d"
    assert space.lower_bound_all_same
    assert not space.upper_bound_all_same
    assert all(space.lower_bound == -1.0)
    assert all(space.upper_bound == np.array([4.0, 2.3, 7.0]))

    a = space.create()
    assert isinstance(a, np.ndarray)
    assert len(a) == 3
    assert a.dtype == space.dtype

    for i in range(len(a)):
        a[i] = space.lower_bound[i] + 0.1

    assert space.to_str(a) == "-0.9;-0.9;-0.9"

    b = space.create()
    assert not space.is_equal(a, b)

    space.copy(b, a)
    assert space.is_equal(a, b)

    b[0] = 2
    assert not space.is_equal(a, b)
    assert space.to_str(b) == "2;-0.9;-0.9"

    with TempFile.create() as path:
        with FileLogger(path) as log, log.key_values("F") as kv:
            space.log_parameters_to(kv)
        result = path.read_all_list()
    assert result == ["BEGIN_F",
                      "name: r3d",
                      "class: moptipy.spaces.vectorspace.VectorSpace",
                      "nvars: 3",
                      "dtype: d",
                      "lb: -1",
                      "ub: 4;2.3;7",
                      "END_F"]

    text = space.to_str(b)
    assert isinstance(text, str)
    assert len(text) > 0

    a = space.from_str(text)
    assert space.is_equal(a, b)

    def _invalid(x) -> np.ndarray:
        x[0] = inf
        return x

    def _valid(x) -> np.ndarray:
        nonlocal space
        return np.clip(x, space.lower_bound, space.upper_bound, x)

    validate_space(space, make_element_invalid=_invalid,
                   make_element_valid=_valid)


def test_vector_space_5() -> None:
    """Test the vector space."""
    lb = np.array([3, 4, 5], float)
    ub = np.array([6, 5.5, 7], float)
    assert len(lb) == len(ub) == 3
    space = VectorSpace(3, lb.copy(), ub.copy())
    assert isinstance(space, Space)
    assert str(space) == "r3d"
    assert all(space.lower_bound == lb)
    assert all(space.upper_bound == ub)
    assert not space.lower_bound_all_same
    assert not space.upper_bound_all_same

    x = 0.5 * (lb + ub)
    space.validate(x)
    x[1] = ub[1] + 0.000001
    with raises(ValueError):
        space.validate(x)

    def _eval(xx: np.ndarray) -> float:
        return float(xx[1])

    assert _eval(x) == x[1]
    assert space.clipped(_eval)(x) == ub[1]

    x[0] = lb[0] - 5
    x[2] = ub[2] + 7

    def _test(xx: np.ndarray) -> float:
        nonlocal space
        space.validate(xx)
        return float(xx[0])

    with raises(ValueError):
        _test(x)

    assert space.clipped(_test)(x) == lb[0]

    x[1] = np.nan
    with raises(ValueError):
        _test(x)

    x[1] = np.nan
    with raises(ValueError):
        space.clipped(_test)(x)
