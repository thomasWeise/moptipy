"""This file shows how a single log file is generated in an experiment."""
from moptipy.algorithms.rls import RLS  # the algorithm we use
from moptipy.examples.jssp.experiment import run_experiment  # the JSSP runner
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle  # 0-ary op
from moptipy.operators.permutations.op1_swap2 import Op1Swap2  # 1-ary op
from moptipy.utils.temp import TempDir  # temp directory tool

# We work in a temporary directory, i.e., delete all generated files on exit.
# For a real experiment, you would put an existing directory path in `td`
# by doing `from moptipy.utils.path import Path; td = Path.directory("mydir")`
# and not use the `with` block.
with TempDir.create() as td:  # create temp directory
    # Execute an experiment consisting of exactly one run.
    # As example domain, we use the job shop scheduling problem (JSSP).
    run_experiment(
        base_dir=td,  # working directory = temp dir
        algorithms=[  # the set of algorithms to use: we use only 1
            # an algorithm is created via a lambda
            lambda inst, pwr: RLS(Op0Shuffle(pwr), Op1Swap2())],
        instances=("demo",),  # use the demo JSSP instance
        n_runs=1,  # perform exactly one run
        n_threads=1)  # use exactly one thread
    # The random seed is automatically generated based on the instance name.
    print(td.resolve_inside(  # so we know algorithm, instance, and seed
        "rls_swap2/demo/rls_swap2_demo_0x5a9363100a272f12.txt")
        .read_all_str())  # read file into string (which then gets printed)
# When leaving "while", the temp dir will be deleted
