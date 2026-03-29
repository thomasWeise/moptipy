"""
Perform 1 run of RLS on OneMax, but log all improving moves in files.

`moptipy` now also allows you to log improving moves into files.
You can provide a base directory and a file name pattern as input to
a so-called `FileImprovementLogger`. Then, every time the optimization
process makes an improving move, a log file is generated with the current
solution.
This feature is useful if you run a particularly long experiment and want to
access intermediate solutions during the run and/or want to make sure to not
lose any result if you abort the experiment early.
Since it is possible that the algorithm may make many many many improving
moves, you can specify a maximum number of log files to retain.
Here we specify 10, meaning that once the eleventh file is generated, the
first file is automatically deleted again.
"""
from pycommons.io.temp import temp_dir

from moptipy.algorithms.so.rls import RLS
from moptipy.api.execution import Execution
from moptipy.api.improvement_logger import FileImprovementLogger
from moptipy.examples.bitstrings.onemax import OneMax
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.operators.bitstrings.op1_flip1 import Op1Flip1
from moptipy.spaces.bitstrings import BitStrings

n = 32  # we chose dimension 32
space = BitStrings(n)  # search in bit strings of length 32
problem = OneMax(n)  # we maximize the number of 1 bits
algorithm = RLS(  # create RLS that
    Op0Random(),  # starts with a random bit string and
    Op1Flip1())  # flips exactly one bit in each step

# We work with a temporary directory which is automatically deleted after this
# experiment.
# For a real experiment, you would put the actual directory for the log files
# as `td` by doing `from pycommons.io.path import Path; td = Path("/my/dir")`
# and not use the `with` block.
with temp_dir() as td:  # create temporary directory `td`
    ex = Execution()  # begin configuring execution
    ex.set_solution_space(space)  # set solution space
    ex.set_objective(problem)  # set objective function
    ex.set_algorithm(algorithm)  # set algorithm
    ex.set_rand_seed(199)  # set random seed to 199
    ex.set_improvement_logger(FileImprovementLogger(td, "result", 10))
    ex.set_max_fes(1000)  # allow at most 1000 function evaluations
    with ex.execute():  # Apply the algorithm to the problem.
        pass  # We do nothing here, just let the experiment run.

    print("\nNow reading and printing all the logged data:")
    for f in sorted(td.list_dir(files=True, directories=False)):
        print(f"====== file {f.basename()} ======")
        print(f.read_all_str())
# The temp file is deleted as soon as we leave the `with` block.
