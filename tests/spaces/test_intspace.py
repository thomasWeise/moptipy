"""Test the integer-string space."""
import numpy as np

# noinspection PyPackageRequirements
from pytest import raises

from moptipy.api import Space
from moptipy.spaces import IntSpace
from moptipy.tests.space import check_space
from moptipy.utils import TempFile, FileLogger


def test_int():
    f = IntSpace(12, 3, 32)
    assert isinstance(f, Space)
    assert f.get_name() == "ints12b3-32"

    a = f.create()
    assert isinstance(a, np.ndarray)
    assert len(a) == 12
    assert a.dtype == f.dtype
    assert all(a == 3)

    for i in range(len(a)):
        a[i] = i
    with raises(ValueError):
        f.validate(a)

    b = f.create()
    assert not f.is_equal(a, b)

    f.copy(b, a)
    assert f.is_equal(a, b)

    b[0] = 5
    assert not f.is_equal(a, b)

    with TempFile() as tmp:
        path = str(tmp)
        with FileLogger(path) as log:
            with log.key_values("F") as kv:
                f.log_parameters_to(kv)
        result = open(path, "r").read().splitlines()
    assert result == ["BEGIN_F",
                      "name: ints12b3-32",
                      "type: <class 'moptipy.spaces.intspace.IntSpace'>",
                      "nvars: 12",
                      "dtype: b",
                      "min: 3",
                      "max: 32",
                      "END_F"]

    text = f.to_str(b)
    assert isinstance(text, str)
    assert len(text) > 0

    a = f.from_str(text)
    assert f.is_equal(a, b)

    check_space(f)
