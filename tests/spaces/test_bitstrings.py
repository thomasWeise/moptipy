from moptipy.spaces import BitStrings
from moptipy.api import Space

from moptipy.utils import TempFile, Logger
import numpy as np


def test_int():
    f = BitStrings(12)
    assert isinstance(f, Space)
    assert f.get_name() == "bits12"

    a = f.x_create()
    assert isinstance(a, np.ndarray)
    assert len(a) == 12
    assert a.dtype == np.dtype(np.bool_)

    for i in range(len(a)):
        a[i] = (i & 1) == 0
    b = f.x_create()
    assert not f.x_is_equal(a, b)

    f.x_copy(a, b)
    assert f.x_is_equal(a, b)

    b[0] = not b[0]
    assert not f.x_is_equal(a, b)

    with TempFile() as tmp:
        path = str(tmp)
        with Logger(path) as log:
            with log.key_values("F") as kv:
                f.log_parameters_to(kv)
        result = open(path, "r").read().splitlines()
    assert result == ["BEGIN_F",
                      "name:bits12",
                      "type:<class 'moptipy.spaces.bitstrings.BitStrings'>",
                      "nvars:12",
                      "END_F"]

    text = f.x_to_str(b)
    assert isinstance(text, str)
    assert len(text) > 0

    a = f.x_from_str(text)
    assert f.x_is_equal(a, b)
