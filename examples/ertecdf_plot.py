"""
We execute 1 algorithm on 11 problems and plot the ERT-ECDFs.

We have already discussed the ECDF plots and ERT plots in the examples
`ecdf_plot.py` and `ert_plot.py`, respectively.
Here we discuss a combination of both chart types: the ERT-ECDF plot.

An Empirical Cumulative Distribution Function (ECDF) plot illustrates the
fraction of runs (executions of an algorithm) that have solved their
corresponding problem instance on the vertical axis, over the runtime that has
been consumed on the horizontal axis.
The (empiricially estimated) Expected Running Time (ERT), on the other hand,
shows how long an algorithm will need (vertical axis) to reach certain goal
objective values (horizontal axis).
The ERT-ECDF plot combines these two concepts to illustrate the fraction of
problem instances (vertical axis) that can be expected to be solved at
specific times (horizontal axis). These times are the ERTs to reach the
optimum.

Whereas an ECDF-plot is based on independent algorithm runs, the ERT-ECDF
is based on all runs per problem instance.
"""
import os
from time import sleep
from webbrowser import open_new_tab

import psutil

from moptipy.algorithms.so.rls import RLS
from moptipy.api.execution import Execution
from moptipy.api.experiment import run_experiment
from moptipy.evaluation.axis_ranger import AxisRanger
from moptipy.evaluation.ertecdf import ErtEcdf
from moptipy.evaluation.plot_ecdf import plot_ecdf
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

# We try to solve all OneMax problems with size 2 to 12
problems = [lambda i=ii: OneMax(i) for ii in range(2, 12)]


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
        Op1Flip1()))  # flip exactly one bit
    ex.set_log_improvements(True)  # log the progress!
    return ex


# We execute the whole experiment in a temp directory.
# For a real experiment, you would put an existing directory path into `td`
# by doing `from moptipy.utils.path import Path; td = Path.directory("mydir")`
# and not use the `with` block.
with TempDir.create() as td:  # create temporary directory `td`
    run_experiment(base_dir=td,  # set the base directory for log files
                   instances=problems,  # define the problem instances
                   setups=[make_rls],  # provide RLS run creator
                   n_runs=21)  # we will execute 21 runs per setup
    # Once we arrived here, the experiment with 11*21 = 231 runs has completed.

    data = []  # we will load the data into this list
    Progress.from_logs(path=td,  # the result directory
                       consumer=data.append,  # put the data into data
                       time_unit="FEs",  # time is in FEs (as opposed to "ms")
                       f_name="plainF")  # use raw, unscaled objective values
    ertecdf = []  # we will load the ERT-ECDFs into this list
    # The below function uses the goal objective values from the log files to
    # compute the ERT-ECDF functions. It groups all runs of one algorithm
    # together and then computes the algorithm's overall ECDF.
    ErtEcdf.from_progresses(data, ertecdf.append)

    # Plot the ERT-ECDF functions.
    # This function will automatically pick the labels of the axes and choose
    # that the horizontal axis (FEs) be log-scaled.
    fig = create_figure(width=4)  # create an empty, 4"-wide figure
    plot_ecdf(ecdfs=ertecdf,  # plot all the data
              figure=fig)  # into the figure
    # Notice that save_figure returns a list of files that has been generated.
    # You can specify multiple formats, e.g., ("svg", "pdf", "png") and get
    # multiple files.
    # Below, we generate a svg image, a pdf image, and a png image and
    # remember the list (containing the generated files) as `files`. We will
    # add all other files we generate to this list and, in the end, display
    # all of them in the web browser.
    files = save_figure(fig=fig,  # store fig to a file
                        file_name="ertecdf_over_log_fes",  # base name
                        dir_name=td,  # store graphic in temporary directory
                        formats=("svg", "png", "pdf"))  # file types
    del fig  # dispose figure

    # Plot the ERT-ECDF functions, but this time do not log-scale the x-axis.
    fig = create_figure(width=4)  # create an empty, 4"-wide figure
    plot_ecdf(ecdfs=ertecdf,  # plot all the data
              figure=fig,  # into the figure
              x_axis=AxisRanger.for_axis("FEs", log_scale=False))
    # This time, we save the image only as svg.
    files.extend(save_figure(fig=fig,  # store fig to a file
                             file_name="ertecdf_over_fes",  # base name
                             dir_name=td,  # store graphic in temp directory
                             formats="svg"))  # file type: this time only svg
    del fig  # dispose figure

    # OK, we have now plotted a set of different ERT-ECDF plots.
    # We will open them in the web browser if we are not in a make build.
    if SHOW_PLOTS_IN_BROWSER:
        for file in files:  # for each file we generated
            open_new_tab(f"file://{file}")  # open a browser tab
        sleep(10)  # sleep 10 seconds (enough time for the browser to load)
# The temp directory is deleted as soon as we leave the `with` block.
# Hence, all the figures generated above as well as the experimental results
# now have disappeared.
