"""
Perform 1 Run of 1 Algorithm*Instance on a simple multi-objective problem.

We implement a simple multi-objective version of the JSSP.
"""
from typing import List

from moptipy.algorithms.mo.morls import MORLS
from moptipy.api.mo_archive import MORecord
from moptipy.api.mo_execution import MOExecution
from moptipy.examples.jssp.gantt_space import GanttSpace
from moptipy.examples.jssp.instance import Instance
from moptipy.examples.jssp.makespan import Makespan
from moptipy.examples.jssp.ob_encoding import OperationBasedEncoding
from moptipy.examples.jssp.worktime import Worktime
from moptipy.mo.problem.weighted_sum import Prioritize
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle
from moptipy.operators.permutations.op1_swapn import Op1SwapN
from moptipy.spaces.permutations import Permutations
from moptipy.utils.nputils import array_to_str
from moptipy.utils.temp import TempFile

instance = Instance.from_resource("orb06")
search_space = Permutations.with_repetitions(  # we will encode solutions as
    instance.jobs, instance.machines)          # permutations with repetitions
solution_space = GanttSpace(instance)  # the space of Gantt charts
encoding = OperationBasedEncoding(instance)
f1 = Makespan(instance)  # the first objective be OneMax
f2 = Worktime(instance)  # the second objective be the exact opposite: ZeroMax
problem = Prioritize(  # when deciding upon single best result, prioritize f1
    [f1, f2])          # over f2
algorithm = MORLS(  # create multi-objective RLS that
    Op0Shuffle(search_space),   # starts with a random permutation and
    Op1SwapN())                 # swaps a random number of elements per step

# We execute the whole experiment in a temp directory.
# For a real experiment, you would put an existing directory path in `td`
# by doing `from moptipy.utils.path import Path; td = Path.directory("mydir")`
# and not use the `with` block.
with TempFile.create() as tf:  # create temporary file `tf`
    ex = MOExecution()  # begin configuring execution
    ex.set_solution_space(solution_space)
    ex.set_search_space(search_space)
    ex.set_encoding(encoding)
    ex.set_objective(problem)  # set the multi-objective problem
    ex.set_algorithm(algorithm)  # set algorithm
    ex.set_rand_seed(199)  # set random seed to 199
    ex.set_log_file(tf)  # set log file = temp file `tf`
    ex.set_max_fes(500)  # allow at most 500 function evaluations
    with ex.execute() as process:  # now run the algorithm*problem combination
        print("We found the following trade-off solutions:")
        print("makespan;worktime")
        arch: List[MORecord] = process.get_archive()
        arch.sort()
        for ae in arch:
            print(array_to_str(ae.fs))

    print("\nNow reading and printing all the logged data:")
    print(tf.read_all_str())  # instead, we load and print the log file
# The temp file is deleted as soon as we leave the `with` block.
