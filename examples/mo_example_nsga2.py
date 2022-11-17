"""
Another example for multi-objective optimization, this time using NSGA-II.

Like in `mo_example.py`, we apply a multi-objective algorithm to a
multi-objective version of the JSSP. This time, however, we use the famous
NSGA-II algorithm instead of the multi-objective version of the randomized
local search. If you compare the results, you will find that NSGA-II indeed
performs much better.
"""

from moptipy.algorithms.mo.nsga2 import NSGA2
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
from moptipy.operators.permutations.op2_gap import (
    Op2GeneralizedAlternatingPosition,
)
from moptipy.spaces.permutations import Permutations
from moptipy.utils.nputils import array_to_str
from moptipy.utils.temp import TempFile

instance = Instance.from_resource("swv02")     # We load the instance "swv02".
solution_space = GanttSpace(instance)          # Solutions are Gantt charts.
search_space = Permutations.with_repetitions(  # We will encode solutions as
    instance.jobs, instance.machines)          # permutations w. repetitions.
encoding = OperationBasedEncoding(instance)    # Decode permutations to Gantt.

# Each multi-objective optimization problem is defined by several objective
# functions *and* a way to scalarize the vector of objective values.
# The scalarization is only used by the system to decide for one single best
# solution in the end *and* if we actually apply a single-objective algorithm
# to the problem instead of a multi-objective one. (Here we will apply a
# multi-objective algorithm.)
f1 = Makespan(instance)  # The first objective be the makespan.
f2 = Worktime(instance)  # The second objective be the total work time.
# Here, we decide for a priorization scalarization: The single best end result
# will be the one with the shortest makespan.
problem = Prioritize([f1, f2])

# NSGA-II is the most well-known multi-objective optimization algorithm.
# It works directly on the multiple objectives. It does not require the
# scalarization above at all. The scalarization is _only_ used internally in
# the `Process` objects to ensure compatibility with single-objective
# optimization and for being able to remember a single "best" solution.
algorithm = NSGA2(              # Create the NSGA-II algorithm.
    Op0Shuffle(search_space),   # start with a random permutation and
    Op1SwapN(),                 # swap a random number of elements per step.
    Op2GeneralizedAlternatingPosition(search_space),  # use this crossover
    16, 0.05)  # population size = 16, crossover rate = 0.05

# We work with a temporary log file which is automatically deleted after this
# experiment. For a real experiment, you would not use the `with` block and
# instead put the path to the file that you want to create into `tf` by doing
# `from moptipy.utils.path import Path; tf = Path.path("mydir/my_file.txt")`.
with TempFile.create() as tf:  # create temporary file `tf`
    ex = MOExecution()  # begin configuring execution
    ex.set_solution_space(solution_space)
    ex.set_search_space(search_space)
    ex.set_encoding(encoding)
    ex.set_objective(problem)      # set the multi-objective problem
    ex.set_algorithm(algorithm)
    ex.set_rand_seed(199)          # set random seed to 199
    ex.set_log_improvements(True)  # log all improving moves
    ex.set_log_file(tf)            # set log file = temp file `tf`
    ex.set_max_fes(2800)           # allow at most 2800 function evaluations
    with ex.execute() as process:  # now run the algorithm*problem combination
        arch: list[MORecord] = process.get_archive()
        print(f"We found the {len(arch)} non-dominated trade-off solutions:")
        print("makespan;worktime")
        arch.sort()
        for ae in arch:
            print(array_to_str(ae.fs))

    print("\nNow reading and printing all the logged data:")
    print(tf.read_all_str())  # instead, we load and print the log file
# The temp file is deleted as soon as we leave the `with` block.
