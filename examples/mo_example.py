"""
A simple example for multi-objective optimization.

We implement a simple multi-objective version of the Job Shop Scheduling
Problem (JSSP). In a JSSP, we have `n` jobs and `m` machines. Each job
needs to be processed by each of the machines in a specific order and needs
a specific time on each of them. Solutions are schedules that assign these
operations to the machines. The time that a schedule needs until all the
operations are completed is called the "makespan." Usually, the goal is to
find the schedule with the shortest possible makespan. However, here we also
consider a second goal: the total "worktime." When a machine receives its
first operation to process, it is turned on. It remains on until it finishes
its last job, after which it is turned off. The time between switching it on
and off be the worktime of the machine and the total "worktime" be the sum of
all of these worktimes.

In our multi-objective version of the JSSP, we want to find schedules that
have both a short makespan and a short worktime. In this situation, there may
be more than one solution: One schedule may have a shorter makespan but a
longer worktime and another one may have a longer worktime and a shorter
makespan. We call such solutions mutually non-dominated. (One solution
dominates another one if it is better in at least one objective and not worse
in all the others.)

In this example, we apply a multi-objective version of the randomized local
search algorithm, `morls`.
"""

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

# MO-RLS is a fully-fledged multi-objective optimization method (though not a
# very good one). It works directly on the multiple objectives. It does not
# require the scalarization above at all. The scalarization is _only_ used
# internally in the `Process` objects to ensure compatibility with single-
# objective optimization and for being able to remember a single "best"
# solution.
algorithm = MORLS(              # Create multi-objective RLS that
    Op0Shuffle(search_space),   # starts with a random permutation and
    Op1SwapN())                 # swaps a random number of elements per step.

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
