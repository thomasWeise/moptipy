"""Test parsing the multi-objective end results."""


from typing import Final, cast

from pycommons.io.path import Path
from pycommons.io.temp import temp_dir
from pycommons.strings.string_conv import str_to_bool

from moptipy.algorithms.mo.nsga2 import NSGA2
from moptipy.api.experiment import run_experiment
from moptipy.api.mo_execution import MOExecution
from moptipy.evaluation.end_results import EndResult
from moptipy.evaluation.mo_end_results import (
    MOEndResult,
    from_csv,
    from_logs,
    to_csv,
)
from moptipy.examples.bitstrings.onemax import OneMax
from moptipy.examples.bitstrings.zeromax import ZeroMax
from moptipy.mo.problem.weighted_sum import Prioritize
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.operators.bitstrings.op1_flip1 import Op1Flip1
from moptipy.operators.bitstrings.op2_uniform import Op2Uniform
from moptipy.spaces.bitstrings import BitStrings


def test_parse_mo_end_result() -> None:
    """Do a single run of RLS on OneMax and parse the result."""
    n_runs: Final[int] = 3
    n: Final[int] = 10
    with temp_dir() as td:
        rd = run_experiment(
            base_dir=td,
            instances=[lambda: Prioritize((OneMax(n), ZeroMax(n)))],
            setups=[lambda om: MOExecution()
                    .set_solution_space(BitStrings(n))
                    .set_algorithm(NSGA2(Op0Random(), Op1Flip1(),
                                         Op2Uniform(), 16, 0.5))
                    .set_max_fes(200)
                    .set_objective(om)],
            n_runs=n_runs,
            perform_warmup=False, perform_pre_warmup=False)
        assert rd == td

        results: list[EndResult] = list(from_logs(td))
        assert list.__len__(results) >= n_runs

        for row in results:
            assert row.x is None
            assert row.y is not None
            x = [str_to_bool(xx) for xx in row.y]
            xn = len(x)
            f = sum(x)
            assert f == cast("MOEndResult", row).fs[1]
            assert (xn - f) == cast("MOEndResult", row).fs[0]

        seeds: set[int] = set()
        for res in results:
            assert isinstance(res, MOEndResult)
            assert tuple.__len__(res.fs) == 2
            for v in res.fs:
                assert isinstance(v, int)
                assert 0 <= v <= n
            seeds.add(res.rand_seed)
        assert set.__len__(seeds) == n_runs

        results.sort()
        csv_file: Final[Path] = td.resolve_inside("csv.txt")
        to_csv(results, csv_file)
        results_2 = list(from_csv(csv_file))
        results_2.sort()
        assert results_2 == results
        for i, a in enumerate(results):
            b = results_2[i]
            assert ((a.x is None) and (b.x is None)) or (a.x == b.x)
            assert a.x is None
            assert b.x is None
            assert a.y is not None
            assert b.y is not None
            assert ((a.y is None) and (b.y is None)) or (a.y == b.y)
