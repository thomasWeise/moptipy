"""Test the bit-string space."""
import numpy as np

from moptipy.api.space import Space
from moptipy.spaces.bitstrings import BitStrings
from moptipy.tests.space import validate_space
from moptipy.utils.logger import FileLogger
from moptipy.utils.temp import TempFile


def test_bit_strings() -> None:
    """Test the bit strings search space."""
    f = BitStrings(12)
    assert isinstance(f, Space)
    assert str(f) == "bits12"
    validate_space(f, make_element_invalid=lambda x: np.array([1]))

    a = f.create()
    assert isinstance(a, np.ndarray)
    assert len(a) == 12
    assert a.dtype == np.dtype(np.bool_)
    assert all(not b for b in a)

    for i in range(len(a)):
        a[i] = (i & 1) == 0
    b = f.create()
    assert not f.is_equal(a, b)

    f.copy(a, b)
    assert f.is_equal(a, b)

    b[0] = not b[0]
    assert not f.is_equal(a, b)

    with TempFile.create() as path:
        with FileLogger(path) as log, log.key_values("F") as kv:
            f.log_parameters_to(kv)
        result = path.read_all_list()
    assert result == ["BEGIN_F",
                      "name: bits12",
                      "class: moptipy.spaces.bitstrings.BitStrings",
                      "nvars: 12",
                      "dtype: ?",
                      "END_F"]

    text = f.to_str(b)
    assert isinstance(text, str)
    assert len(text) > 0

    a = f.from_str(text)
    assert f.is_equal(a, b)
