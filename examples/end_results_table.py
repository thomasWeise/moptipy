r"""
We present the end results of an experiment as a table.

By default, end result tables have two parts. The first part renders
information for each instance-algorithm combination. For each instance, the
statistics are displayed for each algorithm, aggregated over all runs of the
algorithm on an instance. The second part then renders statistics for each
algorithm and aggregated over all runs on all instances.

For both parts, you can choose which statistics should be displayed. The
column headers will automatically be chosen and rendered appropriately. The
default data are equivalent to what we will use in our book:

1. Part 1: Algorithm-Instance statistics
    - `$\instance$`: the instance name
    - `$\lowerBound{\objf}$`: the lower bound of the objective value of the
      instance
    - `setup`: the name of the algorithm or algorithm setup
    - `best`: the best objective value reached by any run on that instance
    - `mean`: the arithmetic mean of the best objective values reached over
       all runs
    - `sd`: the standard deviation of the best objective values reached over
       all runs
    - `mean1`: the arithmetic mean of the best objective values reached over
       all runs, divided by the lower bound (or goal objective value)
    - `mean(FE/ms)`: the arithmetic mean of objective function evaluations
       performed per millisecond, over all runs
    - `mean(t)`: the arithmetic mean of the time in milliseconds when the last
      improving move of a run was applied, over all runs

2. Part 2: Algorithm Summary Statistics
    - `setup`: the name of the algorithm or algorithm setup
    - `best1`: the minimum of the best objective values reached divided by
      the lower bound (or goal objective value) over all runs
    - `gmean1`: the geometric mean of the best objective values reached
       divided by the lower bound (or goal objective value) over all runs
    - `worst1`: the maximum of the best objective values reached divided by
      the lower bound (or goal objective value) over all runs
    - `sd1`: the standard deviation of the best objective values reached
       divided by the lower bound (or goal objective value) over all runs
    - `gmean(FE/ms)`: the geometric mean of objective function evaluations
      performed per millisecond, over all runs
    - `gmean(t)`: the geometric mean of the time in milliseconds when the last
      improving move of a run was applied, over all runs

The best values of each category are always rendered in bold face.
Tables can be rendered in different formats, such as
:py:class:`~moptipy.utils.markdown.Markdown`,
:py:class:`~moptipy.utils.latex.LaTeX`, and
:py:class:`~moptipy.utils.html.HTML`.
"""

from moptipy.algorithms.random_sampling import RandomSampling
from moptipy.algorithms.so.hill_climber import HillClimber
from moptipy.algorithms.so.rls import RLS
from moptipy.api.execution import Execution
from moptipy.api.experiment import run_experiment
from moptipy.evaluation.end_results import EndResult
from moptipy.evaluation.tabulate_end_results import tabulate_end_results
from moptipy.examples.jssp.gantt_space import GanttSpace
from moptipy.examples.jssp.instance import Instance
from moptipy.examples.jssp.makespan import Makespan
from moptipy.examples.jssp.ob_encoding import OperationBasedEncoding
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle
from moptipy.operators.permutations.op1_swap2 import Op1Swap2
from moptipy.spaces.permutations import Permutations
from moptipy.utils.html import HTML
from moptipy.utils.latex import LaTeX
from moptipy.utils.temp import TempDir

# The three JSSP instances we want to try to solve:
problems = [lambda: Instance.from_resource("ft06"),
            lambda: Instance.from_resource("la24"),
            lambda: Instance.from_resource("dmu23")]


def make_rls(problem: Instance) -> Execution:
    """
    Create an RLS Execution.

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
    ex.set_algorithm(RLS(  # create RLS that
        Op0Shuffle(perms),  # create random permutation
        Op1Swap2()))  # swap two jobs
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
            Op1Swap2()))  # swap two jobs
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
# For a real experiment, you would put an existing directory path into `td`
# by doing `from moptipy.utils.path import Path; td = Path.directory("mydir")`
# and not use the `with` block.
with TempDir.create() as td:  # create temporary directory `td`
    run_experiment(base_dir=td,  # set the base directory for log files
                   instances=problems,  # define the problem instances
                   setups=[make_rls,  # provide RLS run creator
                           make_hill_climber,  # provide hill climber
                           make_random_sampling],  # provide random sampling
                   n_runs=7)  # we will execute 31 runs per setup
    # Once we arrived here, the experiment with 3*3*7=63 runs has completed.

    data = []  # we will load the data into this list
    EndResult.from_logs(td, data.append)  # load all end results

    file = tabulate_end_results(data, dir_name=td)  # create the table
    print(f"\nnow presenting markdown data from file {file!r}.\n")
    print(file.read_all_str())  # print the result

    file = tabulate_end_results(data, dir_name=td,
                                text_format_driver=LaTeX.instance)
    print(f"\nnow presenting LaTeX data from file {file!r}.\n")
    print(file.read_all_str())  # print the result

    file = tabulate_end_results(data, dir_name=td,
                                text_format_driver=HTML.instance)
    print(f"\nnow presenting HTML data from file {file!r}.\n")
    print(file.read_all_str())  # print the result

# The temp directory is deleted as soon as we leave the `with` block.
# This means that all log files and the table file have disappeared.
