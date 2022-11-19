"""Test the Gantt space."""
from numpy.random import default_rng

from moptipy.examples.jssp.gantt import Gantt
from moptipy.examples.jssp.gantt_space import GanttSpace
from moptipy.examples.jssp.instance import Instance
from moptipy.examples.jssp.ob_encoding import OperationBasedEncoding
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle
from moptipy.spaces.permutations import Permutations
from moptipy.tests.space import validate_space


def test_gantt_space() -> None:
    """Test the Gantt space."""
    for name in ["abz7", "demo", "dmu48", "la14", "orb05", "swv15", "ta71"]:
        insx = Instance.from_resource(name)

        def __make_valid(x: Gantt, ins=insx) -> Gantt:
            ssp = Permutations.with_repetitions(ins.jobs, ins.machines)
            ob = OperationBasedEncoding(ins)
            op0 = Op0Shuffle(ssp)
            rg = default_rng()
            xx = ssp.create()
            op0.op0(rg, xx)
            ob.decode(xx, x)
            return x

        validate_space(GanttSpace(insx), __make_valid)
