"""Generate an end-results CSV file for an experiment with the JSSP."""
from moptipy.algorithms.ea1plus1 import EA1plus1  # first algo to test
from moptipy.algorithms.hill_climber import HillClimber  # second algo to test
from moptipy.evaluation.end_results import EndResult  # the end result record
from moptipy.examples.jssp.experiment import run_experiment  # JSSP example
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle  # 0-ary op
from moptipy.operators.permutations.op1_swap2 import Op1Swap2  # 1-ary op
from moptipy.utils.temp import TempDir  # tool for temp directories

# We work in a temporary directory, i.e., delete all generated files on exit.
with TempDir.create() as td:
    run_experiment(  # run the JSSP experiment with the following parameters:
        base_dir=td,  # base directory to write all log files to
        algorithms=[  # the set of algorithm generators
            lambda inst, pwr: EA1plus1(Op0Shuffle(pwr), Op1Swap2()),  # algo 1
            lambda inst, pwr: HillClimber(Op0Shuffle(pwr), Op1Swap2())],  # 2
        instances=("demo", "abz7", "la24"),  # we use 3 JSSP instances
        max_fes=10000,  # we grant 10000 FEs per run
        n_runs=4)  # perform 4 runs per algorithm * instance combination

    end_results = []  # this list will receive the end results records
    EndResult.from_logs(td, end_results.append)  # get results from log files

    er_csv = EndResult.to_csv(  # store end results to csv file (returns path)
        end_results,  # the list of end results to store
        td.resolve_inside("end_results.txt"))  # path to the file to generate
    print(er_csv.read_all_str())  # read generated file as string and print it
# When leaving "while", the temp dir will be deleted
