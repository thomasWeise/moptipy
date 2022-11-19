"""Test loading and validity of JSSP instances."""
import numpy as np

from moptipy.examples.jssp.instance import Instance, check_instance


def __check_load_inst(inst: str) -> Instance:
    """Load an instance from a resource and perform basic checks."""
    return check_instance(Instance.from_resource(inst))


def test_load_demo_from_resource() -> None:
    """Test loading the demo instance from resources."""
    i = __check_load_inst("demo")
    assert i.jobs == 4
    assert i.machines == 5
    assert i.makespan_lower_bound == 180
    assert np.array_equal(i, np.array(
        [[[0, 10], [1, 20], [2, 20], [3, 40], [4, 10]],
         [[1, 20], [0, 10], [3, 30], [2, 50], [4, 30]],
         [[2, 30], [1, 20], [4, 12], [3, 40], [0, 10]],
         [[4, 50], [3, 30], [2, 15], [0, 20], [1, 15]]],
        np.int8))


def __check_seq(prefix: str, end: int, start: int = 1, min_len=2) -> None:
    """Load a sequence of instance from resources and perform basic checks."""
    for i in range(start, end + 1):
        s = str(i)
        if len(s) < min_len:
            s = "0" + s
        __check_load_inst(prefix + s)


def test_load_orlib_from_resource() -> None:
    """Check loading the well-known instances."""
    __check_seq("abz", 9, 5, 1)
    __check_seq("dmu", 80)
    __check_load_inst("ft06")
    __check_load_inst("ft10")
    __check_load_inst("ft20")
    __check_seq("la", 40)
    __check_seq("orb", 10)
    __check_seq("swv", 20)
    __check_seq("ta", 80)
    __check_seq("yn", 4, min_len=1)
