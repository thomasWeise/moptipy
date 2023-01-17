"""
Apply 3 algorithms on 6 problems for 21 runs and compare results with tests.

We apply three algorithms to six problem instances and conduct 31 runs per
instance-algorithm combination. We then apply the two-tailed Mann-Whitney U
test pairwise to the results and print its outcomes in a table.

Such a statistical test can tell us how likely the observed difference between
two sets of results would have occurred if the results would stem from the
same algorithm. If this `p`-value is small, then the two sets are likely to
stem from differently-performing algorithms. In order to decide whether or not
a `p`-value is small enough, it is compared to a significance threshold
`alpha`, which is set to `alpha=0.02` by default. In other words, `alpha` is
the maximum probability of "being wrong" that we will accept. However, since
we are performing more than one test, we need to adjust `alpha`. We therefore
use the Bonferroni correction, which sets `alpha'=alpha/n_texts`, where
`n_tests` is the number of conducted tests.

The first column of the table denotes the problem instances. Each of the other
three columns represents a pair of algorithms. In each cell, the pair is
compared based on the results on the instance of the row. The cell ten holds
the `p`-value of the two-tailed Mann-Whitney U test. If the first algorithm
is significantly better (at `p<alpha'`) than the second algorithm, then the
cell is marked with `<`. If the first algorithm is significantly worse (at
`p<alpha'`) than the second algorithm, then the cell is marked with `>`. If
the observed differences are not significant (`p>=alpha'`), then the cell
is marked with `?`.

Finally, the bottom row sums up the numbers of `<`, `?`, and `>` outcomes for
each algorithm pair.

However, there could also be a situation where a statistical comparison makes
no sense as no difference could reliably be detected anyway. For example, if
one algorithm has a smaller median result but a larger mean result, or if the
medians are the same, or if the means are the same. Regardless of what outcome
a test would have, we could not really claim that any of the algorithms was
better or worse. In such cases, no test is performed and `-` is printed
instead (signified by `&mdash;` in the markdown format).
"""
from moptipy.algorithms.so.rls import RLS
from moptipy.api.execution import Execution
from moptipy.api.experiment import run_experiment
from moptipy.evaluation.end_results import EndResult
from moptipy.evaluation.tabulate_result_tests import tabulate_result_tests
from moptipy.examples.bitstrings.leadingones import LeadingOnes
from moptipy.examples.bitstrings.onemax import OneMax
from moptipy.examples.bitstrings.trap import Trap
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.operators.bitstrings.op1_flip1 import Op1Flip1
from moptipy.operators.bitstrings.op1_m_over_n_flip import Op1MoverNflip
from moptipy.spaces.bitstrings import BitStrings
from moptipy.utils.temp import TempDir

# The six problems we want to try to solve:
problems = [lambda: OneMax(100),  # 100-dimensional OneMax
            lambda: OneMax(200),  # 200-dimensional OneMax
            lambda: Trap(100),  # 100-dimensional Trap
            lambda: Trap(200),  # 200-dimensional Trap
            lambda: LeadingOnes(100),  # 100-dimensional LeadingOnes
            lambda: LeadingOnes(200)]  # 200-dimensional LeadingOnes


def rls(problem, op1) -> Execution:
    """
    Create an RLS Execution.

    :param problem: the problem (OneMax, Trap, or LeadingOnes)
    :param op1: the unary operator to use
    :returns: the execution
    """
    return Execution().set_solution_space(BitStrings(problem.n)) \
        .set_objective(problem).set_algorithm(
        RLS(Op0Random(), op1)).set_max_fes(300)


# We execute the whole experiment in a temp directory.
# For a real experiment, you would put an existing directory path into `td`
# by doing `from moptipy.utils.path import Path; td = Path.directory("mydir")`
# and not use the `with` block.
with TempDir.create() as td:  # create temporary directory `td`
    run_experiment(
        base_dir=td,  # set the base directory for log files
        instances=problems,  # define the problem instances
        setups=[lambda p: rls(p, Op1Flip1()),  # RLS + flip 1 bit
                lambda p: rls(p, Op1MoverNflip(p.n, 1)),  # flip bits at p=1/n
                lambda p: rls(p, Op1MoverNflip(p.n, 2)),  # flip bits at p=2/n
                ], n_runs=21)  # conduct 21 independent runs per setup

    # load all the end results
    end_results = []
    EndResult.from_logs(td, end_results.append)

    # create a markdown table with statistical test results
    file = tabulate_result_tests(end_results, dir_name=td)
    print("\n")
    print(file.read_all_str())  # and print it

# The temp directory is deleted as soon as we leave the `with` block.
