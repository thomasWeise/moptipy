"""
We plot the performance of values of a problem feature for one algorithm.

Many benchmark problems have features, such as the problem scale. Sometimes
we want to understand how these features influence the algorithm performance.
With the function :func:`~moptipy.evaluation.\
plot_end_statistics_over_parameter_impl.plot_end_statistics_over_param`, we
can plot one statistic versus the value of a feature (or even algorithm
parameter). In this example, we do this for the number `n` of bits of the
well-known OneMax problem.

Let us say we apply the randomized local search
(:class:`moptipy.algorithms.so.rls.RLS`) with single-bit flip operator to the
well-known :class:`moptipy.examples.bitstrings.onemax.OneMax` problem. The
OneMax is defined over the bit strings of the length `n`.

What influence does this feature `n` have on the time that we need to solve
the problem?

In this example, we test 20 values (`1..20`) for `n` on the minimization
version of the OneMax problem. We apply our RLS to each instance 7 times
and approximate the expected running time (ERT). On the horizontal axis we
put the values of `n`, on the vertical axis, we put the `ERT`.

All what we have to do is tell our system how it can determine the value of
the feature from an "instance setup name" and how to obtain the base name
of the instance. Now that is rather simply: our system names instances
automatically, here according to the form `onemax_n` where `n` is the
feature value. E.g., `onemax_10` means `n=10`. So we can get the value
of `n` as `int(name[7:])` and the base name of the instance as `name[6:]`,
i.e., `onemax` in our example.
"""
import os
from time import sleep
from webbrowser import open_new_tab

import psutil

from moptipy.algorithms.so.rls import RLS
from moptipy.api.execution import Execution
from moptipy.api.experiment import run_experiment
from moptipy.evaluation.axis_ranger import AxisRanger
from moptipy.evaluation.end_results import EndResult
from moptipy.evaluation.end_statistics import EndStatistics
from moptipy.evaluation.plot_end_statistics_over_parameter import (
    plot_end_statistics_over_param,
)
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

# We try to solve 20 onemax instances.
problems = [lambda nn=n: OneMax(nn) for n in range(1, 21)]


def make_rls(problem) -> Execution:
    """
    Create an RLS Execution with 1 bit flip mutation.

    :param problem: the (OneMax) problem
    :returns: the execution
    """
    ex = Execution()
    ex.set_solution_space(BitStrings(problem.n))
    ex.set_objective(problem)
    ex.set_algorithm(RLS(Op0Random(), Op1Flip1()))
    return ex


# We execute the whole experiment in a temp directory.
# For a real experiment, you would put an existing directory path into `td`
# by doing `from moptipy.utils.path import Path; td = Path.directory("mydir")`
# and not use the `with` block.
with TempDir.create() as td:  # create temporary directory `td`
    run_experiment(base_dir=td,  # set the base directory for log files
                   instances=problems,  # define the problem instances
                   setups=[make_rls],  # provide RLS run creator
                   n_runs=7)  # we will execute 7 runs per setup
    # Once we arrived here, the 20*7 = 140 runs have completed.

    end_results = []  # we will load the raw data into this list
    EndResult.from_logs(td, end_results.append)

    end_stats = []  # the end statistics go into this list
    EndStatistics.from_end_results(end_results, end_stats.append)

    files = []  # the collection of files

    # Plot the ERT over the feature `n`.
    fig = create_figure(width=4)  # create an empty, 4"-wide figure
    plot_end_statistics_over_param(
        data=end_stats, figure=fig, y_dim="ertFEs",
        x_getter=lambda es: int(es.instance[7:]),  # => n
        instance_getter=lambda es: es.instance[:6],  # => onemax
        x_label="n", y_axis=AxisRanger())
    # Save the image only as svg and png.
    files.extend(save_figure(fig=fig,  # store fig to a file
                             file_name="ert_over_onemax_n",  # base name
                             dir_name=td,  # store graphic in temp directory
                             formats="svg"))  # file types: only svg
    del fig  # dispose figure

    # OK, we have now generated and saved the plot in a file.
    # We will open it in the web browser if we are not in a make build.
    if SHOW_PLOTS_IN_BROWSER:
        for file in files:  # for each file we generated
            open_new_tab(f"file://{file}")  # open a browser tab
        sleep(10)  # sleep 10 seconds (enough time for the browser to load)
# The temp directory is deleted as soon as we leave the `with` block.
# Hence, all the figures generated above as well as the experimental results
# now have disappeared.
