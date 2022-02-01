"""Test loading and validity of JSSP instances."""
import numpy as np

from moptipy.examples.jssp import Instance


def __check_load_inst(inst: str) -> Instance:
    inst = Instance.from_resource(inst)
    assert isinstance(inst, Instance)
    assert isinstance(inst.machines, int)
    assert inst.machines > 0
    assert isinstance(inst.jobs, int)
    assert inst.jobs > 0
    assert isinstance(inst.makespan_lower_bound, int)
    assert inst.makespan_lower_bound > 0
    assert isinstance(inst.makespan_upper_bound, int)
    assert inst.makespan_upper_bound > inst.makespan_lower_bound
    assert isinstance(inst.matrix, np.ndarray)
    assert inst.matrix.shape[0] == inst.jobs
    assert inst.matrix.shape[1] == 2 * inst.machines
    return inst


def test_load_demo_from_resource():
    """Test loading the demo instance from resources."""
    i = __check_load_inst("demo")
    assert i.jobs == 4
    assert i.machines == 5
    assert i.makespan_lower_bound == 180


def __check_seq(prefix: str, end: int, start: int = 1, min_len=2):
    for i in range(start, end + 1):
        s = str(i)
        if len(s) < min_len:
            s = "0" + s
        __check_load_inst(prefix + s)


def test_load_orlib_from_resource():
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
