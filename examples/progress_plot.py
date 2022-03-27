"""Do an experiment with 2 algorithms on 2 problems and plot the progress."""
import os
from time import sleep
from webbrowser import open_new_tab

import psutil

from moptipy.algorithms.ea1plus1 import EA1plus1
from moptipy.algorithms.random_walk import RandomWalk
from moptipy.api.execution import Execution
from moptipy.api.experiment import run_experiment
from moptipy.evaluation.plot_progress_impl import plot_progress
from moptipy.evaluation.progress import Progress
from moptipy.examples.bitstrings.ising1d import Ising1d
from moptipy.examples.bitstrings.onemax import OneMax
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.operators.bitstrings.op1_m_over_n_flip import Op1MoverNflip
from moptipy.spaces.bitstrings import BitStrings
from moptipy.utils.plot_utils import create_figure, save_figure
from moptipy.utils.temp import TempDir


def show_in_browser() -> bool:
    """
    Check if this script is invoked from a make build.

    :returns: True if the script is invoked from a make build, False
        otherwise.
    """
    prc = psutil.Process(os.getppid())  # get parent process
    while prc is not None:
        if "make" in prc.name():  # is it a make process?
            return False
        prc = prc.parent()  # get next parent
    return True


# The four problems we want to try to solve:
problems = [lambda: OneMax(16),  # 16-dimensional OneMax
            lambda: Ising1d(16)]  # 16-dimensional LeadingOnes


def make_ea1plus1(problem) -> Execution:
    """
    Create a (1+1)-EA Execution.

    :param problem: the problem (OneMax or Ising1d)
    :returns: the execution
    """
    ex = Execution()
    ex.set_solution_space(BitStrings(problem.n))
    ex.set_objective(problem)
    ex.set_algorithm(
        EA1plus1(  # create (1+1)-EA that
            Op0Random(),  # starts with a random bit string and
            Op1MoverNflip(n=problem.n, m=1),  # flips each bit with p=1/n
            op1_is_default=True))  # don't include op1 in algorithm name str
    ex.set_max_fes(100)  # permit 100 FEs
    ex.set_log_improvements(True)
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
            Op1MoverNflip(n=problem.n, m=1),  # flips each bit with p=1/n
            op1_is_default=True))  # don't include op1 in algorithm name str
    ex.set_max_fes(100)
    ex.set_log_improvements(True)
    return ex


# We execute the whole experiment in a temp directory.
# For a real experiment, you would put an existing directory path in `td`
# and not use the `with` block.
with TempDir.create() as td:  # create temporary directory `td`
    rd = td.resolve_inside("results")
    run_experiment(base_dir=rd,  # set the base directory for log files
                   instances=problems,  # define the problem instances
                   setups=[make_ea1plus1,  # provide (1+1)-EA run creator
                           make_random_walk],  # provide HC run creator
                   n_runs=10)  # we will execute 10 runs per setup

    progresses_fes_raw = []
    Progress.from_logs(path=rd,
                       consumer=progresses_fes_raw.append,
                       time_unit="FEs",
                       f_name="plain")

    fig = create_figure()
    plot_progress(progresses=progresses_fes_raw,
                  figure=fig)
    file = save_figure(fig=fig,
                       file_name="single_runs_fes_raw",
                       dir_name=td,
                       formats="svg")[0]
    del fig
    if show_in_browser():
        open_new_tab(f"file://{file}")
        sleep(2)
# The temp directory is deleted as soon as we leave the `with` block.
