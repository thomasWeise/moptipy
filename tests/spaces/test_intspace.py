from moptipy.spaces import IntSpace
from moptipy.api import Space

from moptipy.utils import TempFile, Logger
import numpy as np


def test_int():
    f = IntSpace(12, 3, 32)
    assert isinstance(f, Space)
    assert f.get_name() == "ints12b[3,32]"

    a = f.x_create()
    assert isinstance(a, np.ndarray)
    assert len(a) == 12
    assert a.dtype == f.dtype

    for i in range(len(a)):
        a[i] = i
    b = f.x_create()
    assert not f.x_is_equal(a, b)

    f.x_copy(a, b)
    assert f.x_is_equal(a, b)

    b[0] = 5
    assert not f.x_is_equal(a, b)

    with TempFile() as tmp:
        path = str(tmp)
        with Logger(path) as log:
            with log.key_values("F") as kv:
                f.log_parameters_to(kv)
        result = open(path, "r").read().splitlines()
    assert result == ["BEGIN_F",
                      "name:ints12b[3,32]",
                      "type:<class 'moptipy.spaces.intspace.IntSpace'>",
                      "nvars:12",
                      "dtype:b",
                      "min:3",
                      "max:32",
                      "END_F"]

    text = f.x_to_str(b)
    assert isinstance(text, str)
    assert len(text) > 0

    a = f.x_from_str(text)
    assert f.x_is_equal(a, b)
