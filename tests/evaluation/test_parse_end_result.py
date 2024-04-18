"""Test parsing the end result loader."""

from os import rename

from pycommons.io.temp import temp_dir

from moptipy.algorithms.so.rls import RLS
from moptipy.api.execution import Execution
from moptipy.api.experiment import run_experiment
from moptipy.evaluation.end_results import EndResult, from_logs
from moptipy.examples.bitstrings.onemax import OneMax
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.operators.bitstrings.op1_flip1 import Op1Flip1
from moptipy.spaces.bitstrings import BitStrings


def __check_record(er: EndResult) -> None:
    """
    Check a single end result record with known contents.

    :param er: the record
    """
    assert isinstance(er, EndResult)
    assert er.instance == "onemax_10"
    assert er.algorithm == "rls_flip1"
    assert er.goal_f == 0
    assert er.best_f == 1
    assert er.last_improvement_fe == 8
    assert er.last_improvement_time_millis >= 0
    assert er.max_fes == 20
    assert er.max_time_millis is None
    assert er.rand_seed == 16853959975878339418
    assert er.total_time_millis >= er.last_improvement_time_millis


def test_parse_end_result() -> None:
    """Do a single run of RLS on OneMax and parse the result."""
    with temp_dir() as td:
        rd = run_experiment(
            base_dir=td,
            instances=[lambda: OneMax(10)],
            setups=[lambda om: Execution()
                    .set_solution_space(BitStrings(om.n))
                    .set_algorithm(RLS(Op0Random(), Op1Flip1()))
                    .set_max_fes(20).set_objective(om)],
            n_runs=1, n_threads=1,
            perform_warmup=False, perform_pre_warmup=False)
        assert rd == td
        res_dir_1 = td.resolve_inside("rls_flip1")
        res_dir_1.enforce_dir()
        res_dir_2 = res_dir_1.resolve_inside("onemax_10")
        res_dir_2.enforce_dir()
        result_file = res_dir_2.resolve_inside(
            "rls_flip1_onemax_10_0xe9e54b4d4ce12b5a.txt")
        result_file.enforce_file()

        results: list[EndResult] = []
        for path in [result_file, res_dir_2, res_dir_1, td]:
            results.clear()
            from_logs(path, results.append)
            assert len(results) == 1
            __check_record(results[0])

        # test case-insensitive issues: windows may mangle file name case
        result_file_2 = res_dir_2.resolve_inside(
            "rLs_Flip1_oneMax_10_0xe9e54b4d4ce12b5a.txt")
        rename(result_file, result_file_2)
        for path in [result_file_2, res_dir_2, res_dir_1, td]:
            results.clear()
            from_logs(path, results.append)
            assert len(results) == 1
            __check_record(results[0])
