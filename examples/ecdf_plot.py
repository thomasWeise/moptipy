"""
We execute an experiment with 2 algorithms on 1 problem and plot the ECDFs.

An Empirical Cumulative Distribution Function (ECDF) plot illustrates the
fraction of runs (executions of an algorithm) that have solved their
corresponding problem instance on the vertical axis, over the runtime that has
been consumed on the horizontal axis.
Its highest value is 1, its lowest value is 0.
If you execute 10 runs of your algorithm on two problems each, the ECDF curves
can take on (2*10 + 1) = 21 different values from [0, 1].
If all of your runs succeed in reaching the goal objective value and the
slowest run does so after T2 time units, then the ECDF becomes 1 for time=T2.
The ECDF remains 0 until the time T1 when the fastest run has solved its
corresponding problem instance.
Between T1 and T2, the ECDF is monotonously rising.

If can happen that an ECDF never reaches 1.
For example, in our two-problem situation described above, it could be that
our algorithm can only solve the first problem (all 10 runs succeed), but
never the second problem (all 10 runs fail).
Then, the ECDF will reach 0.5 but no higher value.

Of course, an algorithm is the better, the faster its ECDF rises and the
higher it rises.
In other words, the bigger the area under the ECDF curve, the better.

There are four tools involved in creating such diagrams:

The module `moptipy.utils.plot_utils` provides the functions for allocating
and storing figures (named `create_figure` and `save_figure`).

The module `moptipy.evaluation.progress` includes the class `Progress`.
An instance `Progress` represents a two-dimensional progress curve of a
single run.
Its time dimensions can either be `FEs` or `ms`. In the former case, it
records the time in objective function evaluations (FEs) and in the latter
case, it records them in milliseconds.
Its objective dimension can either be `plainF`, `scaledF` or `normalizedF`,
but for ECDFs, we would usually keep them `plainF`.

The module `moptipy.evaluation.ecdf` includes the class `Ecdf`.
An instance `Ecdf` represents a two-dimensional ECDF curve.
It can be constructed from a selection of `Progress` objects.
Matter of fact, a whole sequence of ECDF curves can automatically be
generated from a sequence of `Progress` objects, which are automatically
grouped by algorithm.

Finally, the module `moptipy.evaluation.plot_ecdf_impl` provides the
function `plot_ecdf`. This function accepts a sequence contain `Ecdf` objects
and illustrates them automatically. It will automatically choose axis labels,
colors, and group and order the data in a way that it deems reasonable.

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
from moptipy.evaluation.ecdf import Ecdf
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

# We try to solve only the 8-bit OneMax problem
problems = [lambda: OneMax(8)]


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
    ex.set_max_fes(256)  # permit 256 FEs
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
    ex.set_max_fes(256)  # permit 256 FEs
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
                   n_runs=31)  # we will execute 31 runs per setup
    # Once we arrived here, the experiment with 2*1*31 = 62 runs has completed.

    data = []  # we will load the data into this list
    Progress.from_logs(path=td,  # the result directory
                       consumer=data.append,  # put the data into data
                       time_unit="FEs",  # time is in FEs (as opposed to "ms")
                       f_name="plainF")  # use raw, unscaled objective values
    ecdf = []  # we will load the ECDFs into this list
    # The below function uses the goal objective values from the log files to
    # compute the ECDF functions. It groups all runs of one algorithm together
    # and then computes the algorithm's overall ECDF.
    Ecdf.from_progresses(data, ecdf.append)

    # Plot the ECDF functions.
    # This function will automatically pick the labels of the axes and choose
    # that the horizontal axis (FEs) be log-scaled.
    fig = create_figure(width=4)  # create an empty, 4"-wide figure
    plot_ecdf(ecdfs=ecdf,  # plot all the data
              figure=fig)  # into the figure
    # Notice that save_figure returns a list of files that has been generated.
    # You can specify multiple formats, e.g., ("svg", "pdf", "png") and get
    # multiple files.
    # Below, we generate a svg image, a pdf image, and a png image and
    # remember the list (containing the generated files) as `files`. We will
    # add all other files we generate to this list and, in the end, display
    # all of them in the web browser.
    files = save_figure(fig=fig,  # store fig to a file
                        file_name="ecdf_over_log_fes",  # base name
                        dir_name=td,  # store graphic in temporary directory
                        formats=("svg", "png", "pdf"))  # file types
    del fig  # dispose figure

    # Plot the ECDF functions, but this time do not log-scale the x-axis.
    fig = create_figure(width=4)  # create an empty, 4"-wide figure
    plot_ecdf(ecdfs=ecdf,  # plot all the data
              figure=fig,  # into the figure
              x_axis=AxisRanger.for_axis("FEs", log_scale=False))
    # This time, we save the image only as svg.
    files.extend(save_figure(fig=fig,  # store fig to a file
                             file_name="ecdf_over_fes",  # base name
                             dir_name=td,  # store graphic in temp directory
                             formats="svg"))  # file type: this time only svg
    del fig  # dispose figure

    # OK, we have now plotted a set of different ECDF plots.
    # We will open them in the web browser if we are not in a make build.
    if SHOW_PLOTS_IN_BROWSER:
        for file in files:  # for each file we generated
            open_new_tab(f"file://{file}")  # open a browser tab
        sleep(10)  # sleep 10 seconds (enough time for the browser to load)
# The temp directory is deleted as soon as we leave the `with` block.
# Hence, all the figures generated above as well as the experimental results
# now have disappeared.
