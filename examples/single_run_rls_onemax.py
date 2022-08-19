"""
Perform 1 Run of 1 Algorithm*Instance combination.

With this example program, we demonstrate how to perform exactly one run of
one algorithm on one optimization. We show how the progress of this run is
stored in a log file and how we can load the text of this log file after the
run.

As example problem, we pick the well-known "OneMax" task from discrete
optimization. Here, the solution space is the set of bit strings of length
`n`. The objective function counts the number of bits that are `1` in the
string. The more bits that are `1`, the better. So if `n=4`, then the optimum
is `1111`. This is a fairly easy problem.

We here apply the RLS to this problem. This algorithm is provided with a
nullary operator that creates a random bit string. It also gets a unary
operator that flips each bit in the bit string independently with probability
`1/n`. We then grant 100 objective function evaluations to this algorithm.

We create a temporary file and execute a single run random seed `199` of the
above algorithm on this problem at `n=10`.
"""
from moptipy.algorithms.so.rls import RLS
from moptipy.api.execution import Execution
from moptipy.examples.bitstrings.onemax import OneMax
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.operators.bitstrings.op1_flip1 import Op1Flip1
from moptipy.spaces.bitstrings import BitStrings
from moptipy.utils.temp import TempFile

n = 10  # we chose dimension 10
space = BitStrings(n)  # search in bit strings of length 10
problem = OneMax(n)  # we maximize the number of 1 bits
algorithm = RLS(  # create RLS that
    Op0Random(),  # starts with a random bit string and
    Op1Flip1())  # flips exactly one bit in each step

# We work with a temporary log file which is automatically deleted after this
# experiment. For a real experiment, you would not use the `with` block and
# instead put the path to the file that you want to create into `tf` by doing
# `from moptipy.utils.path import Path; tf = Path.path("mydir/my_file.txt")`.
with TempFile.create() as tf:  # create temporary file `tf`
    ex = Execution()  # begin configuring execution
    ex.set_solution_space(space)  # set solution space
    ex.set_objective(problem)  # set objective function
    ex.set_algorithm(algorithm)  # set algorithm
    ex.set_rand_seed(199)  # set random seed to 199
    ex.set_log_file(tf)  # set log file = temp file `tf`
    ex.set_max_fes(100)  # allow at most 100 function evaluations
    with ex.execute() as process:  # now run the algorithm*problem combination
        end_result = process.create()  # create empty record to receive result
        process.get_copy_of_best_y(end_result)  # obtain end result
        print(f"Best solution found: {process.to_str(end_result)}")
        print(f"Quality of best solution: {process.get_best_f()}")
        print(f"Consumed Runtime: {process.get_consumed_time_millis()}ms")
        print(f"Total FEs: {process.get_consumed_fes()}")

    print("\nNow reading and printing all the logged data:")
    print(tf.read_all_str())  # instead, we load and print the log file
# The temp file is deleted as soon as we leave the `with` block.
