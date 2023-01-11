"""
We plot the performance of different setups of an algorithm over two problems.

Algorithms and operators have parameters. Sometimes we want to understand how
these parameters influence the algorithm performance. With the function
:func:`~moptipy.evaluation.plot_end_statistics_over_parameter_impl.\
plot_end_statistics_over_param`, we can plot one statistic versus the value of
a parameter (or even instance feature). In this example, we do this for an
algorithm parameter.

Let us say we apply the randomized local search
(:class:`moptipy.algorithms.so.rls.RLS`) to the minimization version of the
well-known :class:`moptipy.examples.bitstrings.leadingones.LeadingOnes`
problem. We need to plug in an unary operator for bit strings. We can choose
the operator :class:`moptipy.operators.bitstrings.op1_m_over_n_flip.\
Op1MoverNflip`, which flips each bit of a bit string with the same
probability, `m/n`, where `n` is the length of the bit string and `m` is a
parameter.

What influence does this parameter have on the solution quality that we can
get after 128 objective function evaluations (FEs)?

In this example, we test eleven values (`1..10`) for `m` on two instances of
LeadingOnes, namely one with `n=16` and one with `n=24` bits. We apply each
algorithm setup 11 times to each instance, totalling `11*11*2=242` runs.

We then plot the mean end result quality for each of the 11 algorithm setups
for each of the two problem instances, as well as the overall mean result
quality over both instances per setup. On the horizontal axis, we place the
values of `m`. On the vertical axis, we put the mean end result quality.

All what we have to do is tell our system how it can determine the value of
the parameter from an "algorithm setup name" and how to obtain the base name
of the algorithm. Now that is rather simply: our system names algorithms
automatically, here according to the form `rls_flipBm` where `m` is the
parameter value. E.g., `rls_flipB10` means `m=10`. So we can get the value
of `m` as `int(name[9:])` and the base name of the algorithm as `name[9:]`,
i.e., `rls_flipB` in our example.
"""
import os
from time import sleep
from webbrowser import open_new_tab

import psutil

from moptipy.algorithms.so.rls import RLS
from moptipy.api.execution import Execution
from moptipy.api.experiment import run_experiment
from moptipy.evaluation.end_results import EndResult
from moptipy.evaluation.end_statistics import EndStatistics
from moptipy.evaluation.plot_end_statistics_over_parameter import (
    plot_end_statistics_over_param,
)
from moptipy.examples.bitstrings.leadingones import LeadingOnes
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.operators.bitstrings.op1_m_over_n_flip import Op1MoverNflip
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

# We try to solve two LeadingOnes instances.
problems = [lambda: LeadingOnes(16), lambda: LeadingOnes(24)]


def make_rls(problem, m: int) -> Execution:
    """
    Create an RLS Execution with `Bin(m/n)` bit flip mutation.

    :param problem: the (LeadingOnes) problem
    :param m: the number of bits to flip, i.e., the parameter
    :returns: the execution
    """
    ex = Execution()
    ex.set_solution_space(BitStrings(problem.n))
    ex.set_objective(problem)
    ex.set_algorithm(RLS(  # create RLS that
        Op0Random(),  # starts with a random bit string and
        Op1MoverNflip(n=problem.n, m=m)))  # bigger m -> more bits to flip
    ex.set_max_fes(128)
    return ex


# the set of algorithm: m ranges from 1 to 10
algorithms = [lambda p, ii=m: make_rls(p, ii) for m in range(1, 11)]

# We execute the whole experiment in a temp directory.
# For a real experiment, you would put an existing directory path into `td`
# by doing `from moptipy.utils.path import Path; td = Path.directory("mydir")`
# and not use the `with` block.
with TempDir.create() as td:  # create temporary directory `td`
    run_experiment(base_dir=td,  # set the base directory for log files
                   instances=problems,  # define the problem instances
                   setups=algorithms,  # provide RLS run creator
                   n_runs=11)  # we will execute 11 runs per setup
    # Once we arrived here, the 11*11*2 = 242 runs have completed.

    end_results = []  # we will load the raw data into this list
    EndResult.from_logs(td, end_results.append)

    end_stats = []  # the end statistics go into this list
    EndStatistics.from_end_results(end_results, end_stats.append)
    EndStatistics.from_end_results(  # over all instances summary
        end_results, end_stats.append, join_all_instances=True)

    files = []  # the collection of files

    # Plot the performance over the parameter `m`.
    fig = create_figure(width=4)  # create an empty, 4"-wide figure
    plot_end_statistics_over_param(
        data=end_stats, figure=fig, y_dim="plainF.mean",
        x_getter=lambda es: int(es.algorithm[9:]),  # => m
        algorithm_getter=lambda es: es.algorithm[:9],  # => rls_flipB
        x_label="m")
    # Save the image only as svg and png.
    files.extend(save_figure(fig=fig,  # store fig to a file
                             file_name="mean_f_over_param",  # base name
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
