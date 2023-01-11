"""
We execute an experiment with 2 algorithms on 1 problem and plot the ERTs.

The (empiricially estimated) Expected Running Time - or ERT for short - gives
us an impression how long an optimization algorithm will need to reach a
certain goal quality for a given optimization problem instance.

If we apply an algorithm once to a problem instance, it will randomly start
at a rather bad solution quality. It will then step by step improve. We can
log each improvement in a log file. If we conduct multiple runs of the
algorithm, we will collect multiple log files from them. For each objective
value `f` that was reached by any of the runs, we can compute the arithmetic
mean that it took them to do so. This works well as long as all runs can
reach `f`. However, especially for better (i.e., small) values of `f`, some
runs may fail to do so. These runs stop improving earlier. The arithmetic
mean of the time to read these `f` would go to infinity.

Luckily, we can do better than that. We can simply assume that once an
unsuccessful run terminates, we would directly start a new run afterwards.
If at least one run in our experiment was successful, this means that the
probability to reach `f` is larger than zero and if we imagine that we
keep starting new runs again and again, we will eventually succeed. This is
the idea behind the ERT. For any goal `f`, it is computed as

  `ERT[f] = Time(fbest >= f) / s`

where `s` is the number of successful runs, i.e., of runs that reached the
goal `f` and `Time(fbest >= f)` is the sum of the runtime of all runs that
was spent until the objective value reached `f` (or the run terminated).

An ERT plot shows the ERT for all objective values `f` encountered during
a set of runs. It is therefore a curve that grows for shrinking `f`.

In this example, we apply both a randomized local search and a random walk
to the minimization version of the 12-bit OneMax problem. We grant each of
them 100 FEs and perform 21 runs per setup. We then plot the ERT in terms
of the expected number of objective function evaluations (FEs) over the
objective values (`f`).
"""
import os
from time import sleep
from webbrowser import open_new_tab

import psutil

from moptipy.algorithms.random_walk import RandomWalk
from moptipy.algorithms.so.rls import RLS
from moptipy.api.execution import Execution
from moptipy.api.experiment import run_experiment
from moptipy.evaluation.axis_ranger import AxisRanger
from moptipy.evaluation.ert import Ert
from moptipy.evaluation.plot_ert import plot_ert
from moptipy.evaluation.progress import Progress
from moptipy.examples.bitstrings.onemax import OneMax
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.operators.bitstrings.op1_flip1 import Op1Flip1
from moptipy.spaces.bitstrings import BitStrings
from moptipy.utils.plot_utils import create_figure, save_figure
from moptipy.utils.temp import TempDir

# We do not show the generated graphics in the browser if this script is
# called from a "make" build. This small lambda checks whether there is any
# process with "make" in its name anywhere in the parent hierarchy of the
# current process.
ns = lambda prc: False if prc is None else (  # noqa: E731
    "make" in prc.name() or ns(prc.parent()))

# should we show the plots?
SHOW_PLOTS_IN_BROWSER = not ns(psutil.Process(os.getppid()))

# We try to solve only the 16-bit OneMax problem
problems = [lambda: OneMax(16)]


def make_rls(problem) -> Execution:
    """
    Create an RLS Execution with single bit flip mutation.

    :param problem: the OneMax problem
    :returns: the execution
    """
    ex = Execution()
    ex.set_solution_space(BitStrings(problem.n))
    ex.set_objective(problem)
    ex.set_algorithm(RLS(  # create RLS that
        Op0Random(),  # starts with a random bit string and
        Op1Flip1()))  # flips exactly one bit
    ex.set_max_fes(100)  # the maximum FEs
    ex.set_log_improvements(True)  # log the progress!
    return ex


def make_random_walk(problem) -> Execution:
    """
    Create a Random Walk Execution.

    :param problem: the OneMax problem
    :returns: the execution
    """
    ex = Execution()
    ex.set_solution_space(BitStrings(problem.n))
    ex.set_objective(problem)
    ex.set_algorithm(
        RandomWalk(  # create a random walk that
            Op0Random(),  # starts with a random bit string and
            Op1Flip1()))  # flip exactly one bit
    ex.set_max_fes(100)  # the maximum FEs
    ex.set_log_improvements(True)  # log the progress!
    return ex


# We execute the whole experiment in a temp directory.
# For a real experiment, you would put an existing directory path into `td`
# by doing `from moptipy.utils.path import Path; td = Path.directory("mydir")`
# and not use the `with` block.
with TempDir.create() as td:  # create temporary directory `td`
    run_experiment(base_dir=td,  # set the base directory for log files
                   instances=problems,  # define the problem instances
                   setups=[make_rls,  # provide RLS run creator
                           make_random_walk],  # provide random walk creator
                   n_runs=21)  # we will execute 71 runs per setup
    # Once we arrived here, the experiment with 2*1*31 = 62 runs has completed.

    data = []  # we will load the data into this list
    Progress.from_logs(path=td,  # the result directory
                       consumer=data.append,  # put the data into data
                       time_unit="FEs",  # time is in FEs (as opposed to "ms")
                       f_name="plainF")  # use raw, unscaled objective values
    ert = []  # we will load the ERT into this list
    # The below function groups all runs of one algorithm and instance
    # together and then computes the ERT.
    Ert.from_progresses(data, ert.append)

    # Plot the ERT functions.
    # This function will automatically pick the labels of the axes and choose
    # that the horizontal axis (FEs) be log-scaled.
    fig = create_figure(width=4)  # create an empty, 4"-wide figure
    plot_ert(erts=ert, figure=fig)  # plot all the data into the figure
    # Notice that save_figure returns a list of files that has been generated.
    # You can specify multiple formats, e.g., ("svg", "pdf", "png") and get
    # multiple files.
    # Below, we generate a svg image, a pdf image, and a png image and
    # remember the list (containing the generated files) as `files`. We will
    # add all other files we generate to this list and, in the end, display
    # all of them in the web browser.
    files = save_figure(fig=fig,  # store fig to a file
                        file_name="log_ert_over_f",  # base name
                        dir_name=td,  # store graphic in temporary directory
                        formats=("svg", "png", "pdf"))  # file types
    del fig  # dispose figure

    # Plot the ECDF functions, but this time do not log-scale the x-axis.
    fig = create_figure(width=4)  # create an empty, 4"-wide figure
    plot_ert(erts=ert, figure=fig,  # plot all the data into the figure
             y_axis=AxisRanger.for_axis("FEs", log_scale=False))
    # This time, we save the image only as svg.
    files.extend(save_figure(fig=fig,  # store fig to a file
                             file_name="ert_over_fes",  # base name
                             dir_name=td,  # store graphic in temp directory
                             formats="svg"))  # file type: this time only svg
    del fig  # dispose figure

    # OK, we have now plotted a set of different ERT plots.
    # We will open them in the web browser if we are not in a make build.
    if SHOW_PLOTS_IN_BROWSER:
        for file in files:  # for each file we generated
            open_new_tab(f"file://{file}")  # open a browser tab
        sleep(10)  # sleep 10 seconds (enough time for the browser to load)
# The temp directory is deleted as soon as we leave the `with` block.
# Hence, all the figures generated above as well as the experimental results
# now have disappeared.
