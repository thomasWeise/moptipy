"""
Perform an experiment with an own algorithm on and problem.

First, we implement our own optimization problem. We choose the problem
"Sort the numbers in a list". The solution space of this problem is the
space of `n` numbers from `0..n-1`. If `n=5`, then the best possible
solution would be `0,1,2,3,4` and the worst possible solution would be
`4,3,2,1,0`. A solution `x` should be rated by computing the number of
"sorting errors" of subsequent numbers, i.e., the number of times that
`x[i]>x[i+1]`. Thus, the solution `0,1,2,3,4` would have objective value
`0` and `4,3,2,1,0` would have objective value `4`. Solution `3,2,5,1,0`
would be rated as `3`.

We implement this objective function by deriving a new class `MySortProblem`
from `Objective`. This class receives a member variable `n` for the number of
elements to sort, the function `evaluate` that counts the number of sorting
errors in a solution, and the optional functions `lower_bound` and
`upper_bound`, which provide the upper and lower limit of possible objective
values. We also override the `__str__` operator to return a short and
descriptive name of the function as to be used in file names.

We then implement a rigid optimization algorithm as class `MyAlgorithm` as a
subclass of `Algorithm`. Here, we only need to override the functions `solve`
and `__str__`.

The function `solve` receives an instance of `Process` which provides all the
information to the search: Its method `create` allows us to create a container
for the solutions (if we use `Permutations` as solution space, then this will
be a `numpy.ndarray`). Its method `copy(a, b)` copies the contents of a
solution `b` to a solution `a`. Its method `should_terminate` becomes `True`
when the algorithm should stop and terminate. Its method `evaluate(x)`
computes the objective value of a given solution (via our objective function,
but it additionally remembers the best-so-far solution). Its function
`get_random` provides a random number generator that has been initialized with
an automatically chosen random seed.

Our `solve` function uses all of this to start by generating a random
permutation. In each iteration, it draws two random indices and swaps the
numbers at them. If the result is better, it will be retained. Otherwise, we
will keep the current solution.

We also implement the `__str__` function for our optimization algorithm, as it
provides the name to be used in file names and directory structures.

With all of these ingredients, we then apply our algorithm to three instances
of our sorting problem: sort 5 numbers, sort 10 numbers, and sort 100 numbers.
Via the experiment execution facility, we apply our algorithm for five runs
to each of these problems. We collect all the results and print them to the
standard out.
"""
from moptipy.api.algorithm import Algorithm
from moptipy.api.execution import Execution
from moptipy.api.experiment import run_experiment
from moptipy.api.objective import Objective
from moptipy.api.process import Process
from moptipy.evaluation.end_results import EndResult
from moptipy.spaces.permutations import Permutations
from moptipy.utils.temp import TempDir


class MySortProblem(Objective):
    """An objective function that rates how well a permutation is sorted."""

    def __init__(self, n: int) -> None:
        """
        Initialize: Set the number of values to sort.

        :param n: the scale of the problem
        """
        super().__init__()
        #: the number of numbers to sort
        self.n = n

    def evaluate(self, x) -> int:
        """
        Compute how often a bigger number follows a smaller one.

        :param x: the permutation
        """
        errors = 0  # we start at zero errors
        for i in range(self.n - 1):  # for i in 0..n-2
            if x[i] > x[i + 1]:  # that's a sorting error!
                errors += 1  # so we increase the number
        return errors  # return result

    def lower_bound(self) -> int:
        """
        Get the lower bound: 0 errors is the optimum.

        Implementing this function is optional, but it can help in two ways:
        First, the optimization processes can be stopped automatically when a
        solution of this quality is reached. Second, the lower bound is also
        checked when the end results of the optimization process are verified.

        :returns: 0
        """
        return 0

    def upper_bound(self) -> int:
        """
        Get the upper bound: n-1 errors is the worst.

        Implementing this function is optional, but it can help, e.g., when
        the results of the optimization process are automatically checked.

        :returns: n-1
        """
        return self.n - 1

    def __str__(self):
        """
        Get the name of this problem.

        This name is used in the directory structure and file names of the
        log files.

        :returns: "sort" + n
        """
        return f"sort{self.n}"


class MyAlgorithm(Algorithm):
    """An example for a simple rigidly structured optimization algorithm."""

    def solve(self, process: Process) -> None:
        """
        Solve the problem encapsulated in the provided process.

        :param process: the process instance which provides random numbers,
            functions for creating, copying, and evaluating solutions, as well
            as the termination criterion
        """
        random = process.get_random()   # get the random number generator
        x_cur = process.create()  # create the record for the current solution
        x_new = process.create()  # create the record for the new solution
        n = len(x_cur)  # get the scale of problem as length of the solution

        x_cur[:] = range(n)  # We start by initializing the initial solution
        random.shuffle(x_cur)  # as [0...n-1] and then randomly shuffle it.
        f_cur = process.evaluate(x_cur)  # compute solution quality

        while not process.should_terminate():  # repeat until we are finished
            process.copy(x_new, x_cur)  # copy current to new solution
            i = random.integers(n)  # choose the first random index
            j = random.integers(n)  # choose the second random index
            x_new[i], x_new[j] = x_new[j], x_new[i]  # swap values at i and j
            f_new = process.evaluate(x_new)  # evaluate the new solution
            if f_new < f_cur:  # if it is better than current solution
                x_new, x_cur = x_cur, x_new  # swap current and new solution
                f_cur = f_new  # and remember quality of new solution

    def __str__(self):
        """
        Get the name of this algorithm.

        This name is then used in the directory path and file name of the
        log files.

        :returns: myAlgo
        """
        return "myAlgo"


# The four problems we want to try to solve:
problems = [lambda: MySortProblem(5),  # sort 5 numbers
            lambda: MySortProblem(10),  # sort 10 numbers
            lambda: MySortProblem(100)]  # sort 100 numbers


def make_execution(problem) -> Execution:
    """
    Create an application of our algorithm to our problem.

    :param problem: the problem (MySortProblem)
    :returns: the execution
    """
    ex = Execution()
    ex.set_solution_space(
        Permutations.standard(problem.n))  # we use permutations of [0..n-1]
    ex.set_objective(problem)  # set the objective function
    ex.set_algorithm(MyAlgorithm())  # apply our algorithm
    ex.set_max_fes(100)  # permit 100 FEs
    return ex


# We execute the whole experiment in a temp directory.
# For a real experiment, you would put an existing directory path into `td`
# by doing `from moptipy.utils.path import Path; td = Path.directory("mydir")`
# and not use the `with` block.
with TempDir.create() as td:  # create temporary directory `td`
    run_experiment(base_dir=td,  # set the base directory for log files
                   instances=problems,  # define the problem instances
                   setups=[make_execution],  # creator for our algorithm
                   n_runs=5)  # we will execute 5 runs per setup

    EndResult.from_logs(  # parse all log files and print end results
        td, lambda er: print(f"{er.algorithm} on {er.instance}: {er.best_f}"))
# The temp directory is deleted as soon as we leave the `with` block.
