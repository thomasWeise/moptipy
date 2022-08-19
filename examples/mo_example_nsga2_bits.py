"""
An example for multi-objective optimization with NSGA-II.

This time, we use two obviously contradicting objectives defined over the
space of bit strings: First, we want to maximize the number of `0`-bits in
the strings. Second, we want to maximize the length of the continuous
sequence of `1` bits at the beginning of the string. Hence, for a bit string
of length `n`, we can have `n+1` non-dominated solutions, including the
all-`0` and the all-`1` string.
"""

from moptipy.algorithms.mo.nsga2 import NSGA2
from moptipy.api.mo_execution import MOExecution
from moptipy.examples.bitstrings.leadingones import LeadingOnes
from moptipy.examples.bitstrings.zeromax import ZeroMax
from moptipy.mo.problem.weighted_sum import WeightedSum
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.operators.bitstrings.op1_flip1 import Op1Flip1
from moptipy.operators.bitstrings.op2_uniform import Op2Uniform
from moptipy.spaces.bitstrings import BitStrings
from moptipy.utils.temp import TempFile

solution_space = BitStrings(16)  # We search a bit string of length 16,
f1 = ZeroMax(16)                 # that has as many 0s in it as possible
f2 = LeadingOnes(16)             # and the longest leading sequence of 1s.
# These are, of course, two conflicting goals.
# Each multi-objective optimization problem is defined by several objective
# functions *and* a way to scalarize the vector of objective values.
# The scalarization is only used by the system to decide for one single best
# solution in the end *and* if we actually apply a single-objective algorithm
# to the problem instead of a multi-objective one. (Here we will apply a
# multi-objective algorithm, though.)
# Here, we decide for a weighted sum scalarization, weighting the number of
# zeros half as much as the number of leading ones.
problem = WeightedSum([f1, f2], [1, 2])

# NSGA-II is the most well-known multi-objective optimization algorithm.
# It works directly on the multiple objectives. It does not require the
# scalarization above at all. The scalarization is _only_ used internally in
# the `Process` objects to ensure compatibility with single-objective
# optimization and for being able to remember a single "best" solution.
algorithm = NSGA2(  # Create the NSGA-II algorithm.
    Op0Random(),    # start with a random bit string and
    Op1Flip1(),     # flips single bits as mutation
    Op2Uniform(),   # performs uniform crossover
    10, 0.05)  # population size = 10, crossover rate = 0.05

# We work with a temporary log file which is automatically deleted after this
# experiment. For a real experiment, you would not use the `with` block and
# instead put the path to the file that you want to create into `tf` by doing
# `from moptipy.utils.path import Path; tf = Path.path("mydir/my_file.txt")`.
with TempFile.create() as tf:  # create temporary file `tf`
    ex = MOExecution()  # begin configuring execution
    ex.set_solution_space(solution_space)
    ex.set_objective(problem)      # set the multi-objective problem
    ex.set_algorithm(algorithm)
    ex.set_rand_seed(200)          # set random seed to 200
    ex.set_log_improvements(True)  # log all improving moves
    ex.set_log_file(tf)            # set log file = temp file `tf`
    ex.set_max_fes(300)            # allow at most 300 function evaluations
    with ex.execute():             # now run the algorithm*problem combination
        pass

    print("\nNow reading and printing all the logged data:")
    print(tf.read_all_str())  # instead, we load and print the log file
# The temp file is deleted as soon as we leave the `with` block.
