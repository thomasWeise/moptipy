from moptipy.spaces import VectorSpace
from moptipy.api import Space

from moptipy.utils import TempFile, Logger
import numpy as np


def test_vectors():
    f = VectorSpace(12)
    assert isinstance(f, Space)
    assert f.get_name() == "vector12d"

    a = f.create()
    assert isinstance(a, np.ndarray)
    assert len(a) == 12
    assert a.dtype == f.dtype

    for i in range(len(a)):
        a[i] = i
    b = f.create()
    assert not f.is_equal(a, b)

    f.copy(a, b)
    assert f.is_equal(a, b)

    b[0] = 5
    assert not f.is_equal(a, b)

    with TempFile() as tmp:
        path = str(tmp)
        with Logger(path) as log:
            with log.key_values("F") as kv:
                f.log_parameters_to(kv)
        result = open(path, "r").read().splitlines()
    assert result == ["BEGIN_F",
                      "name:vector12d",
                      "type:<class 'moptipy.spaces.vectorspace.VectorSpace'>",
                      "nvars:12",
                      "dtype:d",
                      "END_F"]

    text = f.to_str(b)
    assert isinstance(text, str)
    assert len(text) > 0

    a = f.from_str(text)
    assert f.is_equal(a, b)
