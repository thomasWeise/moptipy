"""
We plot the end results of an experiment with 2 algorithms on 3 problems.

End results plots are violin plots, which show the distribution of the
data samples for values, overlaid with Box Plots, which show the raw
statistical information in a straightforward way. Both plots are vertical.

A violin plot shows you the approximate distribution of the data.
It is aligned vertically, meaning, e.g., that the vertical axis of the plot is
the best achieved objective value.
For any objective value along the vertical axis, the width of violin plot then
corresponds to how often the value was encountered.

A box plot shows similar information, but boiled down to quantiles and
outliers.
It is therefore more concrete and easier to compare visually. However, it does
not really show you whether the distribution of values is uni- or multi-modal,
which, in turn, is visible in the violin plots.

We here therefore conduct a small example experiment, collect its end results,
and plot them into such a violin-box plot.

See Also
- https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.violinplot.html
- https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.boxplot.html
"""
from time import sleep
from webbrowser import open_new_tab

from pycommons.io.temp import temp_dir

from moptipy.algorithms.so.hill_climber import HillClimber
from moptipy.algorithms.so.rls import RLS
from moptipy.api.execution import Execution
from moptipy.api.experiment import run_experiment
from moptipy.evaluation.end_results import EndResult
from moptipy.evaluation.plot_end_results import plot_end_results
from moptipy.examples.jssp.gantt_space import GanttSpace
from moptipy.examples.jssp.instance import Instance
from moptipy.examples.jssp.makespan import Makespan
from moptipy.examples.jssp.ob_encoding import OperationBasedEncoding
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle
from moptipy.operators.permutations.op1_swap2 import Op1Swap2
from moptipy.spaces.permutations import Permutations
from moptipy.utils.plot_utils import create_figure, save_figure
from moptipy.utils.sys_info import is_make_build

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
    ex.set_max_fes(200)  # permit 200 FEs
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
    ex.set_max_fes(200)  # permit 200 FEs
    return ex


# We execute the whole experiment in a temp directory.
# For a real experiment, you would put an existing directory path into `td`
# by doing `from pycommons.io.path import Path; td = directory_path("mydir")`
# and not use the `with` block.
with temp_dir() as td:  # create temporary directory `td`
    run_experiment(base_dir=td,  # set the base directory for log files
                   instances=problems,  # define the problem instances
                   setups=[make_rls,  # provide RLS run creator
                           make_hill_climber],  # provide hill climber
                   n_runs=31)  # we will execute 31 runs per setup
    # Once we arrived here, the experiment with 2*3*31=186 runs has completed.

    data = []  # we will load the data into this list
    EndResult.from_logs(td, data.append)  # load all end results

    # Plot the end results in a scaled fashion: All objective values are
    # divided by the lower bound. Therefore, 1 is optimal.
    fig = create_figure(width=4)  # create an empty, 4"-wide figure
    plot_end_results(end_results=data,
                     figure=fig,
                     dimension="scaledF")
    # Notice that save_figure returns a list of files that has been generated.
    # You can specify multiple formats, e.g., ("svg", "pdf", "png") and get
    # multiple files.
    # Below, we only generate one svg image and remember the list (containing
    # the single generated file) as `files`. We will add all other files we
    # generate to this list and, in the end, display all of them in the web
    # browser.
    files = save_figure(fig=fig,  # store fig to a file
                        file_name="end_results_scaled",  # base name
                        dir_name=td,  # store graphic in temporary directory
                        formats="svg")  # file type = svg
    del fig  # dispose figure

    # Plot the end results in an un-scaled fashion: All objective values are
    # used as is. This makes the problems less comparable.
    fig = create_figure(width=4)  # create an empty, 4"-wide figure
    plot_end_results(end_results=data, figure=fig, dimension="plainF")
    files.extend(save_figure(fig=fig,  # store fig to a file
                             file_name="end_results",  # base name
                             dir_name=td,  # store graphic in temp directory
                             formats="svg"))  # file type = svg
    del fig  # dispose figure

    # Now we compare the FE when the last improvement happened.
    fig = create_figure(width=4)  # create an empty, 4"-wide figure
    plot_end_results(end_results=data, figure=fig,
                     dimension="lastImprovementFE")
    files.extend(save_figure(fig=fig,  # store fig to a file
                             file_name="last_improv_fe",  # base name
                             dir_name=td,  # store graphic in temp directory
                             formats="svg"))  # file type = svg
    del fig  # dispose figure

    # OK, we have now plotted a set of different end results plots.
    # We will open them in the web browser if we are not in a make build.
    if not is_make_build():
        for file in files:  # for each file we generated
            open_new_tab(f"file://{file}")  # open a browser tab
        sleep(10)  # sleep 10 seconds (enough time for the browser to load)
# The temp directory is deleted as soon as we leave the `with` block.
# Hence, all the figures generated above as well as the experimental results
# now have disappeared.
