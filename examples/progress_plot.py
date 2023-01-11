"""
We execute an experiment with 2 algorithms on 2 problems and plot the progress.

Progress diagrams illustrate how algorithms, well, progress over time.
They can be created from log files if these include the improvements, i.e,
were created with `ex.set_log_improvements(True)`.

There are four tools involved in creating such diagrams:

The module `moptipy.utils.plot_utils` provides the functions for allocating
and storing figures (named `create_figure` and `save_figure`).

The module `moptipy.evaluation.progress` includes the class `Progress`.
An instance `Progress` represents a two-dimensional progress curve of a
single run.
Its time dimensions can either be `FEs` or `ms`. In the former case, it
records the time in objective function evaluations (FEs) and in the latter
case, it records them in milliseconds.
Its objective dimension can either be `plainF`, `scaledF` or `normalizedF`.
In the first case, all objective values `f` are represented as recorded in the
log file. In the second case, they are divided by the goal objective value
`goal_f` (as recorded in the log file) and in the last case, they are
normalized as `(f - goal_f) / goal_f`. The former two cases will fail if the
log file does not contain goal objective values. But if they do, this allows
you to show progress plots of different problems in the same plot in a
reasonable way.

The module `moptipy.evaluation.stat_run` provides the class `StatRun`. This
class allows you to convert a set of `Progress` objects into a statistics
curve. For example, maybe you have collected 10 runs of an optimization
algorithm for a given problem. You do not want to plot all 10 runs, but
instead, you want to plot the arithmetic mean of their objective values. You
can then construct a statistics run for these objects and plot that one
instead.

Finally, the module `moptipy.evaluation.plot_progress_impl` provides the
function `plot_progress`. This function accepts a sequence contain `Progress`
and/or `StatRun` objects and illustrates them automatically. It will
automatically choose axis labels, colors, and group and order the data in
a way that it deems reasonable.

In this file, we will use all the above tools.
We will run a small experiment, parse the resulting log files, and then
illustrate different groupings and selections of the data.
We will create svg figures and open them in the web browser for viewing.
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
from moptipy.evaluation.plot_progress import plot_progress
from moptipy.evaluation.progress import Progress
from moptipy.evaluation.stat_run import StatRun
from moptipy.examples.bitstrings.ising1d import Ising1d
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

# The two problems we want to try to solve:
problems = [lambda: OneMax(32),  # 32-dimensional OneMax
            lambda: Ising1d(32)]  # 32-dimensional one-dimensional Ising Model


def make_rls(problem) -> Execution:
    """
    Create an RLS Execution.

    :param problem: the problem (OneMax or Ising1d)
    :returns: the execution
    """
    ex = Execution()
    ex.set_solution_space(BitStrings(problem.n))
    ex.set_objective(problem)
    ex.set_algorithm(RLS(  # create RLS that
        Op0Random(),  # starts with a random bit string and
        Op1Flip1()))  # flip exactly one bit
    ex.set_max_fes(200)  # permit 200 FEs
    ex.set_log_improvements(True)  # log the progress!
    return ex


def make_random_walk(problem) -> Execution:
    """
    Create a Random Walk Execution.

    :param problem: the problem (OneMax or Ising1d)
    :returns: the execution
    """
    ex = Execution()
    ex.set_solution_space(BitStrings(problem.n))
    ex.set_objective(problem)
    ex.set_algorithm(
        RandomWalk(  # create a random walk that
            Op0Random(),  # starts with a random bit string and
            Op1Flip1()))  # flip exactly one bit
    ex.set_max_fes(200)  # permit 200 FEs
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
                   n_runs=5)  # we will execute 5 runs per setup
    # Once we arrived here, the experiment with 2*2*5 = 20 runs has completed.

    data = []  # we will load the data into this list
    Progress.from_logs(path=td,  # the result directory
                       consumer=data.append,  # put the data into data
                       time_unit="FEs",  # time is in FEs (as opposed to "ms")
                       f_name="plainF")  # use raw, unscaled objective values

    # The first plot will contain every single one of the 20 runs.
    # The system will choose different styles for different algorithms
    # and different problems.
    # It will also automatically pick the labels of the axes and choose
    # that the horizontal axis (FEs) be log-scaled.
    fig = create_figure(width=4)  # create an empty, 4"-wide figure
    plot_progress(progresses=data,  # plot all the data
                  figure=fig)  # into the figure
    # Notice that save_figure returns a list of files that has been generated.
    # You can specify multiple formats, e.g., ("svg", "pdf", "png") and get
    # multiple files.
    # Below, we only generate one svg image and remember the list (containing
    # the single generated file) as `files`. We will add all other files we
    # generate to this list and, in the end, display all of them in the web
    # browser.
    files = save_figure(fig=fig,  # store fig to a file
                        file_name="progress_single_runs_f_over_log_fes",
                        dir_name=td,  # store graphic in temporary directory
                        formats="svg")  # file type = svg
    del fig  # dispose figure

    # Let us now only plot the runs on the OneMax problem.
    # This will still use both algorithms.
    # The system will again choose styles to make the curves distinguishable
    # and will log-scale the horizontal axes.
    fig = create_figure(width=4)  # create an empty, 4"-wide figure
    plot_progress(progresses=[r for r in data  # plot selection of data
                              if ("onemax" in r.instance)],  # only OneMax
                  figure=fig)  # plot into figure fig
    # We save the figure as a file and add this file to the list `files`.
    files.extend(save_figure(
        fig=fig,  # plot into figure fig
        file_name="progress_single_runs_f_over_log_fes_onemax",
        dir_name=td,  # store file in temporary directory
        formats="svg"))  # again as svg
    del fig  # dispose fig

    # We now plot only the runs of the RLS, but for both problems.
    # The system will again choose styles to make the curves distinguishable
    # and will log-scale the horizontal axes.
    fig = create_figure(width=4)  # create an empty, 4"-wide figure
    plot_progress(progresses=[r for r in data if ("rls" in r.algorithm)],
                  figure=fig)
    # This time, we save three files: a svg, a pdf, and a png. Later all
    # three will be opened in the web browser.
    files.extend(save_figure(
        fig=fig,
        file_name="progress_single_runs_f_over_log_fes_rls",
        dir_name=td, formats=("svg", "pdf", "png")))
    del fig

    # Let us now convert the progress data to statistics runs.
    # We apply StatRun.from_progress to a copy of the data and append all
    # new StatRuns to the original data list.
    # This function will automatically choose to compute statistics over
    # algorithm*instance combinations unless we tell it otherwise.
    # We tell it to compute the arithmetic means.
    StatRun.from_progress(source=list(data),  # iterate over _copy_ of data
                          statistics="mean",  # compute the mean f over FEs
                          consumer=data.append)  # and store to data list
    fig = create_figure(width=4)  # create a 4"-wide, empty figure
    # We now plot the single runs AND the mean result quality over time into
    # the same diagram. Notice that the system will again automatically choose
    # an appropriate style.
    plot_progress(progresses=data, figure=fig)
    files.extend(save_figure(
        fig=fig,  # save the figure
        file_name="progress_single_runs_and_mean_f_over_log_fes",
        dir_name=td, formats="svg"))
    del fig  # dispose figure

    fig = create_figure(width=4)  # create a 4"-wide, empty figure
    # We now create the same plot again, but this time we do not log-scale
    # the horizontal (FEs) axis.
    plot_progress(progresses=data, figure=fig,
                  x_axis=AxisRanger.for_axis("FEs", log_scale=False))
    files.extend(save_figure(
        fig=fig, file_name="progress_single_runs_and_mean_f_over_fes",
        dir_name=td, formats=("pdf", "svg")))
    del fig  # dispose figure

    fig = create_figure(width=4)  # create an empty, 4"-wide figure
    # This time we only plot the arithmetic mean runs.
    # We generate a pdf and a svg.
    plot_progress(progresses=[d for d in data if isinstance(d, StatRun)],
                  figure=fig,
                  x_axis=AxisRanger.for_axis("FEs", log_scale=False))
    files.extend(save_figure(fig=fig,
                             file_name="progress_mean_f_over_fes",
                             dir_name=td, formats=("pdf", "svg")))
    del fig  # dispose figure

    # OK, we have now plotted a set of different progress plots.
    # We will open them in the web browser if we are not in a make build.
    if SHOW_PLOTS_IN_BROWSER:
        for file in files:  # for each file we generated
            open_new_tab(f"file://{file}")  # open a browser tab
        sleep(10)  # sleep 10 seconds (enough time for the browser to load)
# The temp directory is deleted as soon as we leave the `with` block.
# Hence, all the figures generated above as well as the experimental results
# now have disappeared.
