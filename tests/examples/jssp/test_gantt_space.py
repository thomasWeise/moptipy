"""Test the Gantt space."""

from numpy.random import default_rng

import moptipy.tests.space as sp
from moptipy.examples.jssp.gantt import Gantt
from moptipy.examples.jssp.gantt_space import GanttSpace
from moptipy.examples.jssp.instance import Instance
from moptipy.examples.jssp.ob_encoding import OperationBasedEncoding
from moptipy.operators.pwr.op0_shuffle import Op0Shuffle
from moptipy.spaces.permutationswr import PermutationsWithRepetitions


def test_gantt_space():
    """Test the Gantt space."""
    for name in ["abz7", "demo", "dmu48", "la14", "orb05", "swv15", "ta71"]:
        insx = Instance.from_resource(name)

        def __make_valid(x: Gantt, ins=insx) -> Gantt:
            ssp = PermutationsWithRepetitions(ins.jobs, ins.machines)
            ob = OperationBasedEncoding(ins)
            op0 = Op0Shuffle(ssp)
            rg = default_rng()
            xx = ssp.create()
            op0.op0(rg, xx)
            ob.map(xx, x)
            return x

        sp.test_space(GanttSpace(insx), __make_valid)
