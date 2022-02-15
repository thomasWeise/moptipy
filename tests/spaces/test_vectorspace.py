"""Test the unconstrained real vector space."""
import numpy as np

import moptipy.tests.space as tst
from moptipy.api.space import Space
from moptipy.spaces.vectorspace import VectorSpace
from moptipy.utils.logger import FileLogger
from moptipy.utils.temp import TempFile


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

    with TempFile.create() as path:
        with FileLogger(path) as log:
            with log.key_values("F") as kv:
                f.log_parameters_to(kv)
        result = path.read_all_list()
    assert result == ["BEGIN_F",
                      "name: vector12d",
                      "class: moptipy.spaces.vectorspace.VectorSpace",
                      "nvars: 12",
                      "dtype: d",
                      "END_F"]

    text = f.to_str(b)
    assert isinstance(text, str)
    assert len(text) > 0

    a = f.from_str(text)
    assert f.is_equal(a, b)

    tst.test_space(f)
