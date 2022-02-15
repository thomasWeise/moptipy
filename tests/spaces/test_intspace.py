"""Test the integer-string space."""
import numpy as np
# noinspection PyPackageRequirements
from pytest import raises

import moptipy.tests.space as tst
from moptipy.api.space import Space
from moptipy.spaces.intspace import IntSpace
from moptipy.utils.logger import FileLogger
from moptipy.utils.temp import TempFile


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

    for i in range(len(a)):
        a[i] = int(f.min_value + (i % (f.max_value - f.min_value)))
    f.validate(a)

    b = f.create()
    assert not f.is_equal(a, b)

    f.copy(b, a)
    assert f.is_equal(a, b)

    b[0] = 5
    assert not f.is_equal(a, b)

    with TempFile.create() as path:
        with FileLogger(path) as log:
            with log.key_values("F") as kv:
                f.log_parameters_to(kv)
        result = path.read_all_list()
    assert result == ["BEGIN_F",
                      "name: ints12b3-32",
                      "class: moptipy.spaces.intspace.IntSpace",
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

    tst.test_space(f)
