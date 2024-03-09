"""
The end results of an algorithm on 3 problems at different stages.

We here demonstrate how we can take "snapshots" of the current end results of
runs at arbitrary time limits. Our log files can collect the whole progress
that a run makes over time. This allows us to plot the final end results, as
we do in example ``end_results_plot.py`. However, if we indeed collect the
whole progress of a run, we can also analyze the results that we would get if
we would stop the runs earlier. Our `EndResults` log file parser allows to
specify arbitrary cut-off times, either specified in terms of objective
function evaluations (FEs) or in milliseconds, as well as cut-off qualities.

The end results plots here are violin plots, which show the distribution of
the data samples for values, overlaid with Box Plots, which show the raw
statistical information in a straightforward way. Both plots are vertical.
See `end_results_plot.py` for details.
"""
from time import sleep
from webbrowser import open_new_tab

from pycommons.io.temp import temp_dir

from moptipy.algorithms.so.rls import RLS
from moptipy.api.execution import Execution
from moptipy.api.experiment import run_experiment
from moptipy.evaluation.axis_ranger import AxisRanger
from moptipy.evaluation.end_results import EndResult
from moptipy.evaluation.plot_end_results import plot_end_results
from moptipy.examples.bitstrings.onemax import OneMax
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.operators.bitstrings.op1_flip1 import Op1Flip1
from moptipy.spaces.bitstrings import BitStrings
from moptipy.utils.plot_utils import (
    create_figure_with_subplots,
    label_box,
    save_figure,
)
from moptipy.utils.sys_info import is_make_build

# The three OneMax instances we want to try to solve:
problems = [lambda: OneMax(10), lambda: OneMax(20), lambda: OneMax(30)]


def make_rls(problem: OneMax) -> Execution:
    """
    Create an RLS Execution.

    :param problem: the OneMax instance
    :returns: the execution
    """
    ex = Execution()  # create execution context
    ex.set_solution_space(BitStrings(problem.n))  # set search space
    ex.set_objective(problem)  # objective function
    ex.set_algorithm(RLS(Op0Random(), Op1Flip1()))  # create RLS
    ex.set_max_fes(128)  # permit 128 FEs
    ex.set_log_improvements(True)  # log improvements <- important here!
    return ex


# We execute the whole experiment in a temp directory.
# For a real experiment, you would put an existing directory path into `td`
# by doing `from pycommons.io.path import Path; td = directory_path("mydir")`
# and not use the `with` block.
with temp_dir() as td:  # create temporary directory `td`
    run_experiment(base_dir=td,  # set the base directory for log files
                   instances=problems,  # define the problem instances
                   setups=[make_rls],  # provide RLS run creator
                   n_runs=31)  # we will execute 31 runs per setup
# Once we arrived here, the experiment with 3*31=93 runs has completed.

# We now collect the end results at different stages, namely at 16 FEs,
# 32 FEs, 64 FEs, and after the full rumtime = 128 FEs.
    data_16 = []  # results after 16 FEs
    EndResult.from_logs(td, data_16.append, max_fes=16)  # load end results
    data_32 = []  # results after 32 FEs
    EndResult.from_logs(td, data_32.append, max_fes=32)  # load end results
    data_64 = []  # results after 64 FEs
    EndResult.from_logs(td, data_64.append, max_fes=64)  # load end results
    data_128 = []  # results after 128 FEs
    EndResult.from_logs(td, data_128.append, max_fes=128)  # load end results
    items = [[16, data_16], [32, data_32], [64, data_64], [128, data_128]]

# We create a multi-figure, i.e., one figure with multiple charts inside.
# The system will automatically assign our plot items to sub-plots.
    fig, plots = create_figure_with_subplots(
        items=len(items), max_rows=4, max_cols=2, max_items_per_plot=1,
        default_height_per_row=2)
    for axes, item, _, _, _, _ in plots:
        plot_end_results(end_results=items[item][1],
                         figure=axes,
                         dimension="bestF",
                         y_axis=AxisRanger(chosen_min=0.0, chosen_max=15.0))
        label_box(axes, f"maxFEs={items[item][0]}", 0.5, 1.0)
# Notice that save_figure returns a list of files that has been generated.
# You can specify multiple formats, e.g., ("svg", "pdf", "png") and get
# multiple files. Below, we generate one svg image and one `png` image and
# remember the list of `files`.
    files = save_figure(fig=fig,  # store fig to a file
                        file_name="selected_end_results",  # base name
                        dir_name=td,  # store graphic in temporary directory
                        formats=("svg", "png"))  # file type = svg and png
    del fig  # dispose figure

# OK, we have now plotted a set of different end results plots at different
# times. We will open them in the web browser if we are not in a make build.
    if not is_make_build():
        for file in files:  # for each file we generated
            open_new_tab(f"file://{file}")  # open a browser tab
        sleep(10)  # sleep 10 seconds (enough time for the browser to load)
# The temp directory is deleted as soon as we leave the `with` block.
# Hence, all the figures generated above as well as the experimental results
# now have disappeared.
