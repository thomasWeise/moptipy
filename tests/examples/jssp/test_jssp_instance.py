from moptipy.examples.jssp import JSSPInstance
from numpy import ndarray


def test_load_demo_from_resource():
    i = JSSPInstance.from_resource("demo")
    assert isinstance(i, JSSPInstance)
    assert i.jobs == 4
    assert i.machines == 5
    assert i.makespan_lower_bound == 180
    assert isinstance(i.matrix, ndarray)


def test_load_orlib_from_resource():
    i = JSSPInstance.from_resource("abz5")
    assert isinstance(i, JSSPInstance)
    assert i.jobs > 0
    assert i.machines > 0
    assert i.makespan_lower_bound > 0
    assert isinstance(i.matrix, ndarray)

    i = JSSPInstance.from_resource("dmu33")
    assert isinstance(i, JSSPInstance)
    assert i.jobs > 0
    assert i.machines > 0
    assert i.makespan_lower_bound > 0
    assert isinstance(i.matrix, ndarray)

    i = JSSPInstance.from_resource("ft10")
    assert isinstance(i, JSSPInstance)
    assert i.jobs > 0
    assert i.machines > 0
    assert i.makespan_lower_bound > 0
    assert isinstance(i.matrix, ndarray)

    i = JSSPInstance.from_resource("la10")
    assert isinstance(i, JSSPInstance)
    assert i.jobs > 0
    assert i.machines > 0
    assert i.makespan_lower_bound > 0
    assert isinstance(i.matrix, ndarray)

    i = JSSPInstance.from_resource("orb04")
    assert isinstance(i, JSSPInstance)
    assert i.jobs > 0
    assert i.machines > 0
    assert i.makespan_lower_bound > 0
    assert isinstance(i.matrix, ndarray)

    i = JSSPInstance.from_resource("swv12")
    assert isinstance(i, JSSPInstance)
    assert i.jobs > 0
    assert i.machines > 0
    assert i.makespan_lower_bound > 0
    assert isinstance(i.matrix, ndarray)

    i = JSSPInstance.from_resource("ta23")
    assert isinstance(i, JSSPInstance)
    assert i.jobs > 0
    assert i.machines > 0
    assert i.makespan_lower_bound > 0
    assert isinstance(i.matrix, ndarray)

    i = JSSPInstance.from_resource("yn4")
    assert isinstance(i, JSSPInstance)
    assert i.jobs > 0
    assert i.machines > 0
    assert i.makespan_lower_bound > 0
    assert isinstance(i.matrix, ndarray)
