"""Test the operation-based encoding for the JSSP."""
import numpy.random as rnd

from moptipy.examples.jssp.gantt_space import GanttSpace
from moptipy.examples.jssp.instance import JSSPInstance
# from moptipy.examples.jssp.gantt import Gantt
from moptipy.examples.jssp.ob_encoding import OperationBasedEncoding
from moptipy.spaces.permutationswr import PermutationsWithRepetitions
from moptipy.tests.encoding import check_encoding
from moptipy.tests.space import check_space


def __check_for_instance(instance: str,
                         random: rnd.Generator = rnd.default_rng()):
    inst = JSSPInstance.from_resource(instance)

    x_space = PermutationsWithRepetitions(inst.jobs, inst.machines)
    check_space(x_space)

    y_space = GanttSpace(inst)
    check_space(y_space, make_valid=None)

    g = OperationBasedEncoding(inst)
    check_encoding(g, x_space, y_space)

    x = x_space.create()
    x_space.validate(x)

    y = y_space.create()
    g.map(x, y)
    y_space.validate(y)

    random.shuffle(x)
    g.map(x, y)
    y_space.validate(y)


def __check_seq(prefix: str, end: int, start: int = 1,
                random: rnd.Generator = rnd.default_rng(),
                min_len: int = 2):
    for i in random.choice(range(start, end + 1), 2):
        s = str(i)
        if len(s) < min_len:
            s = "0" + s
        __check_for_instance(prefix + s, random=random)


def test_for_selected():
    random: rnd.Generator = rnd.default_rng()

    __check_seq("abz", 9, 5, random=random, min_len=1)
    __check_seq("dmu", 80, random=random)
    __check_for_instance("ft06")
    __check_for_instance("ft10")
    __check_for_instance("ft20")
    __check_seq("la", 40, random=random)
    __check_seq("orb", 10, random=random)
    __check_seq("swv", 20, random=random)
    __check_seq("ta", 80, random=random)
    __check_seq("yn", 4, random=random, min_len=1)
