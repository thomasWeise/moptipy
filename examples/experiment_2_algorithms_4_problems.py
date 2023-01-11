"""
Perform an experiment with 2 algorithms on 4 problems for 5 runs.

We want to execute a complete experiment where we apply two algorithms to
four problems and perform five runs per algorithm * problem combination.
As problems, we pick OneMax at dimensions 10 and 32 as well as LeadingOnes at
dimensions 10 and 32. As algorithms, we choose the RLS and simple random
sampling.

We need to provide functions that instantiate the problems and the algorithms.
The experiment execution framework will first instantiate the problem and
thus, e.g., create an instance of OneMax at dimension 10. Then, it will pass
this instance to the function that instantiates the algorithm. This function
could then allocate the right solution space, allocate the right operators,
instantiate the algorithm, and feed all of this into an `Execution` object.
It can also set limits such as a maximum number of permitted function
evaluations in that object (which we do, we allow up to 100 FEs).

The experiment execution framework will automatically and reproducible choose
the random seed for each run. It will also create a directory structure of the
form `algorithm/instance/algorithm_instance_seed.txt` for the log files. It
will automatically use multi-processing, if multiple CPUs are available.
It will also perform warmup-runs, e.g., to make sure that all Python code has
properly been loaded and prepared before executing the actual experiment.

So with the code below, we can generate a structured experiment. We here
set a temporary directory as root folder for everything and then load the
end results from the log files and print them to standard out.
"""
from moptipy.algorithms.random_sampling import RandomSampling
from moptipy.algorithms.so.rls import RLS
from moptipy.api.execution import Execution
from moptipy.api.experiment import run_experiment
from moptipy.evaluation.end_results import EndResult
from moptipy.examples.bitstrings.leadingones import LeadingOnes
from moptipy.examples.bitstrings.onemax import OneMax
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.operators.bitstrings.op1_flip1 import Op1Flip1
from moptipy.spaces.bitstrings import BitStrings
from moptipy.utils.temp import TempDir

# The four problems we want to try to solve:
problems = [lambda: OneMax(10),  # 10-dimensional OneMax
            lambda: OneMax(32),  # 32-dimensional OneMax
            lambda: LeadingOnes(10),  # 10-dimensional LeadingOnes
            lambda: LeadingOnes(32)]  # 32-dimensional LeadingOnes


def make_rls(problem) -> Execution:
    """
    Create an RLS Execution.

    :param problem: the problem (OneMax or LeadingOnes)
    :returns: the execution
    """
    ex = Execution()
    ex.set_solution_space(BitStrings(problem.n))
    ex.set_objective(problem)
    ex.set_algorithm(RLS(  # create RLS that
        Op0Random(),  # starts with a random bit string and
        Op1Flip1()))  # flips one bit in each step
    ex.set_max_fes(100)  # permit 100 FEs
    return ex


def make_random_sampling(problem) -> Execution:
    """
    Create a Random Sampling Execution.

    :param problem: the problem (OneMax or LeadingOnes)
    :returns: the execution
    """
    ex = Execution()
    ex.set_solution_space(BitStrings(problem.n))
    ex.set_objective(problem)
    ex.set_algorithm(RandomSampling(Op0Random()))
    ex.set_max_fes(100)
    return ex


# We execute the whole experiment in a temp directory.
# For a real experiment, you would put an existing directory path into `td`
# by doing `from moptipy.utils.path import Path; td = Path.directory("mydir")`
# and not use the `with` block.
with TempDir.create() as td:  # create temporary directory `td`
    run_experiment(base_dir=td,  # set the base directory for log files
                   instances=problems,  # define the problem instances
                   setups=[make_rls,  # provide RLS run creator
                           make_random_sampling],  # provide RS run creator
                   n_runs=5)  # we will execute 5 runs per setup

    EndResult.from_logs(  # parse all log files and print end results
        td, lambda er: print(f"{er.algorithm} on {er.instance}: {er.best_f}"))
# The temp directory is deleted as soon as we leave the `with` block.
