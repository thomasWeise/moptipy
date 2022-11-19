"""Test the simple cache."""
from moptipy.utils.cache import is_new


def test_is_new() -> None:
    """Test the is_new function."""
    is_new_1 = is_new()
    assert is_new_1 is not None
    is_new_2 = is_new()
    assert is_new_2 is not None
    assert is_new_1 is not is_new_2

    assert is_new_1("a")
    assert is_new_2("a")

    assert is_new_1("b")
    assert is_new_1("c")
    assert is_new_2("c")

    assert not is_new_1("a")
    assert not is_new_2("c")
    assert not is_new_1("b")
    assert not is_new_1("c")

    assert is_new_2("b")
    assert not is_new_2("b")

    is_new_3 = is_new()
    assert is_new_3("b")
    assert is_new_3("c")
    assert is_new_3("a")
    assert not is_new_3("b")
    assert not is_new_3("c")
    assert not is_new_3("a")
