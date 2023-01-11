"""
An example showing how a single log file is generated in an experiment.

As basis for the experiment, we use the Job Shop Scheduling Problem (JSSP)
and apply a simple Randomized Local Search (RLS) to it. We only perform one
single run on one single instance, the trivial `demo` instance. Afterwards,
we print the contents of the log file to the console. We also load the Gantt
chart that was the result of the experiment from the log file and print it,
too - just for fun.
"""
from moptipy.algorithms.so.rls import RLS  # the algorithm we use
from moptipy.examples.jssp.experiment import run_experiment  # the runner
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle  # 0-ary op
from moptipy.operators.permutations.op1_swap2 import Op1Swap2  # 1-ary op
from moptipy.utils.temp import TempDir  # temp directory tool

# We work with a temporary log file which is automatically deleted after this
# experiment. For a real experiment, you would not use the `with` block and
# instead put the path to the file that you want to create into `tf` by doing
# `from moptipy.utils.path import Path; tf = Path.path("mydir/my_file.txt")`.
with TempDir.create() as td:  # create temp directory
    # Execute an experiment consisting of exactly one run.
    # As example domain, we use the job shop scheduling problem (JSSP).
    run_experiment(
        base_dir=td,  # working directory = temporary directory
        algorithms=[  # the set of algorithms to use: we use only 1
            # an algorithm is created via a lambda
            lambda inst, pwr: RLS(Op0Shuffle(pwr), Op1Swap2())],
        instances=("demo",),  # use the demo JSSP instance
        n_runs=1)  # perform exactly one run
    # The random seed is automatically generated based on the instance name.
    file = td.resolve_inside(  # so we know algorithm, instance, and seed
        "rls_swap2/demo/rls_swap2_demo_0x5a9363100a272f12.txt")
    print(file.read_all_str())  # read file into string and print contents

# When leaving "while", the temp directory will be deleted
