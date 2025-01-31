"""Get an end-result statistics CSV file for an experiment with the JSSP."""
from pycommons.io.temp import temp_dir  # tool for temp directories

from moptipy.algorithms.so.hill_climber import HillClimber  # second algo
from moptipy.algorithms.so.rls import RLS  # first algo to test
from moptipy.evaluation.end_results import from_logs  # the end result record
from moptipy.evaluation.end_statistics import from_end_results, to_csv
from moptipy.examples.jssp.experiment import run_experiment  # JSSP example
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle  # 0-ary op
from moptipy.operators.permutations.op1_swap2 import Op1Swap2  # 1-ary op

# We work in a temporary directory, i.e., delete all generated files on exit.
# For a real experiment, you would put an existing directory path into `td` by
# doing `from pycommons.io.path import Path; td = Path("mydir")` and not use
# the `with` block.
with temp_dir() as td:
    run_experiment(  # run the JSSP experiment with the following parameters:
        base_dir=td,  # base directory to write all log files to
        algorithms=[  # the set of algorithm generators
            lambda inst, pwr: RLS(Op0Shuffle(pwr), Op1Swap2()),  # algo 1
            lambda inst, pwr: HillClimber(Op0Shuffle(pwr), Op1Swap2())],  # 2
        instances=("demo", "abz7", "la24"),  # we use 3 JSSP instances
        max_fes=10000,  # we grant 10000 FEs per run
        n_runs=4)  # perform 4 runs per algorithm * instance combination

    # Compute the end statistics from end results loaded from log files.
    end_stats = list(from_end_results(from_logs(td)))

    # store the statistics to a CSV file
    es_csv = to_csv(end_stats, td.resolve_inside("end_stats.txt"))
    print(es_csv.read_all_str())  # read and print the file
# When leaving "while", the temp directory will be deleted
