"""
Perform an experiment with 2 algorithms on 2 problems for 3 runs.

We want to execute a complete experiment where we apply two algorithms to
two problems and perform three runs per algorithm * problem combination.
As problems, we pick OneMax and LeadingOnes at dimension 64.
As algorithms, we choose the RLS and simple random sampling.

Different from `experiment_2_algorithms_4_problems.py`, we this time log
the solutions created during all improving moves to separate files.
This allows us to start a potentially very very long experiment and take
glimpses at the current best solutions *while the experiment is running.*
Of course, an algorithm may make very many improving moves.
Therefore, the `FileImprovementLoggerFactory` allows us to specify a
maximum number of intermediate log files to retain for each algorithm run.
We here set this number to 5, meaning that if there are more than 5 improving
moves, the system will automatically delete the oldest logs.
"""
from pycommons.io.temp import temp_dir

from moptipy.algorithms.random_sampling import RandomSampling
from moptipy.algorithms.so.rls import RLS
from moptipy.api.execution import Execution
from moptipy.api.experiment import run_experiment
from moptipy.api.improvement_logger import FileImprovementLoggerFactory
from moptipy.examples.bitstrings.leadingones import LeadingOnes
from moptipy.examples.bitstrings.onemax import OneMax
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.operators.bitstrings.op1_flip1 import Op1Flip1
from moptipy.spaces.bitstrings import BitStrings

# The four problems we want to try to solve:
problems = [lambda: OneMax(64),  # 64-dimensional OneMax
            lambda: LeadingOnes(64)]  # 64-dimensional LeadingOnes

# The factory for improvement loggers.
# Setting max_files=5 permits at most five log files.
logger_factory = FileImprovementLoggerFactory(max_files=5)


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
    ex.set_max_fes(1024)  # permit 1024 FEs
    ex.set_improvement_logger(logger_factory)
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
    ex.set_max_fes(1024)
    ex.set_improvement_logger(logger_factory)
    return ex


# We execute the whole experiment in a temp directory.
# For a real experiment, you would put an existing directory path into `td` by
# doing `from pycommons.io.path import Path; td = Path("mydir")` and not use
# the `with` block.
with temp_dir() as td:  # create temporary directory `td`
    run_experiment(base_dir=td,  # set the base directory for log files
                   instances=problems,  # define the problem instances
                   setups=[make_rls,  # provide RLS run creator
                           make_random_sampling],  # provide RS run creator
                   n_runs=3)  # we will execute 3 runs per setup

    # The structure is td/algorithm/problemInstance/run/intermediate log.
    for d0 in sorted(td.list_dir(files=False, directories=True)):
        for d1 in sorted(d0.list_dir(files=False, directories=True)):
            for d2 in sorted(d1.list_dir(files=False, directories=True)):
                for f in sorted(d2.list_dir(files=True, directories=False)):
                    print(f"============= {f.relative_to(td)} =============")
                    print(f.read_all_str())
# The temp directory is deleted as soon as we leave the `with` block.
