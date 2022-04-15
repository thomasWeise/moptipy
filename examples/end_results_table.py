"""We present the end results as a table."""

from moptipy.algorithms.ea1plus1 import EA1plus1
from moptipy.algorithms.hill_climber import HillClimber
from moptipy.algorithms.random_sampling import RandomSampling
from moptipy.api.execution import Execution
from moptipy.api.experiment import run_experiment
from moptipy.evaluation.end_results import EndResult
from moptipy.evaluation.tabulate_end_results_impl import tabulate_end_results
from moptipy.examples.jssp.gantt_space import GanttSpace
from moptipy.examples.jssp.instance import Instance
from moptipy.examples.jssp.makespan import Makespan
from moptipy.examples.jssp.ob_encoding import OperationBasedEncoding
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle
from moptipy.operators.permutations.op1_swap2 import Op1Swap2
from moptipy.spaces.permutations import Permutations
from moptipy.utils.temp import TempDir

# The three JSSP instances we want to try to solve:
problems = [lambda: Instance.from_resource("ft06"),
            lambda: Instance.from_resource("la24"),
            lambda: Instance.from_resource("dmu23")]


def make_ea1plus1(problem: Instance) -> Execution:
    """
    Create a (1+1)-EA Execution.

    :param problem: the JSSP instance
    :returns: the execution
    """
    ex = Execution()  # create execution context
    perms = Permutations.with_repetitions(  # create search space as
        problem.jobs, problem.machines)  # permutations with repetitions
    ex.set_search_space(perms)  # set search space
    ex.set_encoding(OperationBasedEncoding(problem))  # set encoding
    ex.set_solution_space(GanttSpace(problem))  # solution space: Gantt charts
    ex.set_objective(Makespan(problem))  # objective function is makespan
    ex.set_algorithm(  # now construct algorithm
        EA1plus1(  # create (1+1)-EA that
            Op0Shuffle(perms),  # create random permutation
            Op1Swap2(),  # swap two jobs
            op1_is_default=True))  # don't include op1 in algorithm name str
    ex.set_max_time_millis(10)  # permit 10 ms of runtime
    return ex


def make_hill_climber(problem: Instance) -> Execution:
    """
    Create a hill climber Execution.

    :param problem: the JSSP instance
    :returns: the execution
    """
    ex = Execution()  # create execution context
    perms = Permutations.with_repetitions(  # create search space as
        problem.jobs, problem.machines)  # permutations with repetitions
    ex.set_search_space(perms)  # set search space
    ex.set_encoding(OperationBasedEncoding(problem))  # set encoding
    ex.set_solution_space(GanttSpace(problem))  # solution space: Gantt charts
    ex.set_objective(Makespan(problem))  # objective function is makespan
    ex.set_algorithm(  # now construct algorithm
        HillClimber(  # create hill climber that
            Op0Shuffle(perms),  # create random permutation
            Op1Swap2(),  # swap two jobs
            op1_is_default=True))  # don't include op1 in algorithm name str
    ex.set_max_time_millis(10)  # permit 10 ms of runtime
    return ex


def make_random_sampling(problem: Instance) -> Execution:
    """
    Create a random sampling Execution.

    :param problem: the JSSP instance
    :returns: the execution
    """
    ex = Execution()  # create execution context
    perms = Permutations.with_repetitions(  # create search space as
        problem.jobs, problem.machines)  # permutations with repetitions
    ex.set_search_space(perms)  # set search space
    ex.set_encoding(OperationBasedEncoding(problem))  # set encoding
    ex.set_solution_space(GanttSpace(problem))  # solution space: Gantt charts
    ex.set_objective(Makespan(problem))  # objective function is makespan
    ex.set_algorithm(  # now construct algorithm
        RandomSampling(  # create hill climber that
            Op0Shuffle(perms)))  # create random permutation
    ex.set_max_time_millis(10)  # permit 10 ms of runtime
    return ex


# We execute the whole experiment in a temp directory.
# For a real experiment, you would put an existing directory path in `td`
# by doing `from moptipy.utils.path import Path; td = Path.directory("mydir")`
# and not use the `with` block.
with TempDir.create() as td:  # create temporary directory `td`
    run_experiment(base_dir=td,  # set the base directory for log files
                   instances=problems,  # define the problem instances
                   setups=[make_ea1plus1,  # provide (1+1)-EA run creator
                           make_hill_climber,  # provide hill climber
                           make_random_sampling],  # provide random sampling
                   n_runs=7,  # we will execute 31 runs per setup
                   n_threads=1)  # we use only a single thread here
    # Once we arrived here, the experiment with 3*3*7=63 runs has completed.

    data = []  # we will load the data into this list
    EndResult.from_logs(td, data.append)  # load all end results

    file = tabulate_end_results(data, dir_name=td)  # create the table
    print(f"\nnow loading table data from file '{file}'.\n")
    print(file.read_all_str())  # print the result

# The temp directory is deleted as soon as we leave the `with` block.
# This means that all log files and the table file have disappeared.
