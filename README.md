[![make build](https://github.com/thomasWeise/moptipy/actions/workflows/build.yaml/badge.svg)](https://github.com/thomasWeise/moptipy/actions/workflows/build.yaml)

# moptipy: Metaheuristic Optimization in Python

This will be a library with implementations of metaheuristic optimization methods in Python.
Well, it will eventually be, because I first need to learn Python.

- [Introduction](#1-introduction)
- [Installation](#2-installation)
- [How-Tos](#3-how-tos)
  - [Applying 1 Algorithm Once to 1 Problem](#31-how-to-apply-1-optimization-algorithm-once-to-1-problem-instance)
  - [Run a Series of Experiments](#32-how-to-run-a-series-of-experiments)
  - [How to Solve and Optimization Problem](#33-how-to-solve-an-optimization-problem)
    - [Defining a New Problem](#331-define-a-new-problem-type)
    - [Defining a New Algorithm](#332-define-a-new-algorithm)
    - [Applying an Own Algorithm to an Own Problem](#333-applying-an-own-algorithm-to-an-own-problem)
- [Data Formats](#4-data-formats)
  - [Log Files](#41-log-files)
  - [End Results CSV Files](#42-end-result-csv-files)
  - [End Result Statistics CSV Files](#43-end-result-statistics-csv-files)
- [Evaluating Experiments](#5-evaluating-experiments)
  - [Exporting Data](#51-exporting-data)
  - [Progress Plots](#52-progress-plots)
  - [ECDF Plots](#53-ecdf-plots)
  - [End Result Plots](#54-end-results-plot)
- [License](#6-license)
- [Contact](#7-contact)


## 1. Introduction

Metaheuristic optimization algorithms are methods for solving hard problems.
Here we provide an API that can be used to implement them and to experiment with them.

A metaheuristic algorithm can be a black-box method, which can solve problems without any knowledge about their nature.
Such a black-box algorithm only requires methods to create points in the search space and to evaluate their quality.
With these operations, it will try to step-by-step discover better points.
Black-box metaheuristics are very general and can be adapted to almost any optimization problem.
White and gray-box algorithms, on the other hand, are tailored to specified problems.
They have more knowledge about these problems.
They make use of the problem structure and can implement more efficient search operations.

Within our `moptipy` framework, you can implement all of these algorithms under a unified [API](https://thomasweise.github.io/moptipy/moptipy.api.html).
What `moptipy` also offers is an [experiment execution facility](https://thomasweise.github.io/moptipy/moptipy.api.html#module-moptipy.api.experiment) that can gather detailed [log information](#4-data-formats) and [evaluate](#5-evaluating-experiments) the gathered results.

The codes in this repository are used as examples in the book [Optimization Algorithms](https://thomasweise.github.io/oa/) which I am currently writing.
Its full sources are available on GitHub at <https://github.com/thomasWeise/oa>.


## 2. Installation

In order to use this package and to, e.g., run the example codes, you need to first install it using `pip`.
You can install the newest version of this library using `pip` by doing

```shell
pip install git+https://github.com/thomasWeise/moptipy.git
```

Alternatively, if you have set up a private/public key for GitHub, you can also do:

```shell
git clone ssh://git@github.com/thomasWeise/moptipy
git install moptipy
```

This may sometimes work better if you are having trouble reaching GitHub via `https` or `http`.
You can also clone the repository and then run a `make` build, which will automatically install all dependencies, run all the tests, and then install the package on your system, too.
If this build completes successful, you can be sure that `moptipy` will work properly on your machine.


## 3. How-Tos

You can find many examples of how to use the [moptipy](https://thomasweise.github.io/moptipy) library in the folder `examples`.
Here, we talk mainly about directly applying one or multiple optimization algorithm(s) to one or multiple optimization problem instance(s).
In [Section 4 on Data Formats](#4-data-formats), we give examples and specifications of the log files that our system produces and how you can export the data to other formats.
Later, in [Section 5 on Evaluating Experiments](#5-evaluating-experiments), we provide several examples on how to evaluate and visualize the results of experiments.

### 3.1. How to Apply 1 Optimization Algorithm Once to 1 Problem Instance

The most basic task that we can do in the domain of optimization is to apply one algorithm to one instance of an optimization problem.
In our framework, we refer to this as an "execution."
You can prepare an execution using the class [`Execution`](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.execution.Execution) in the module [moptipy.api.execution](https://thomasweise.github.io/moptipy/moptipy.api.html#module-moptipy.api.execution).
This class follows the [builder design pattern](https://python-patterns.guide/gang-of-four/builder/).
A builder is basically an object that allows you to step-by-step set the parameters of another, more complicated object that should be created.
Once you have set all parameters, you can create the object.
In our case, the class [`Execution`](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.execution.Execution) allows you to compose all the elements necessary for the algorithm run and then it performs it and provides you the end results of that execution.

So first, you create an instance `ex` of [`Execution`](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.execution.Execution).
Then you set the [algorithm](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.algorithm.Algorithm) that should be applied via the method [`ex.set_algorithm(...)`](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.execution.Execution.set_algorithm).
Then you set the [objective function](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.objective.Objective) via the method [`ex.set_objective(...)`](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.execution.Execution.set_objective).

Then, via [`ex.set_solution_space(...)`](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.execution.Execution.set_solution_space) you set the solution [space](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.space.Space) that contains all possible solutions and is explored by the algorithm.
The solution space is an instance of the class [`Space`](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.space.Space).
It provides all methods necessary to create a solution data structure, to copy the contents of one solution data structure to another one, to convert solution data structures to and from strings, and to verify whether a solution data structure is valid.
It is used by the optimization algorithm for instantiating the solution data structures and for copying them.
It is used internally by the `moptipy` system to automatically maintain copies of the current best solution, to check if the solutions are indeed valid once the algorithm finishes, and to convert the solution to a string to store it in the [log files](#41-log-files).

If the search and solution spaces are different, then you can also set a search [space](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.space.Space) via [`ex.set_search_space(...)`](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.execution.Execution.set_search_space) and an [encoding](https://thomasweise.github.io/moptipy/moptipy.api.html#module-moptipy.api.encoding) via [`ex.set_encoding(...)`](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.execution.Execution.set_encoding).
This is not necessary if the algorithm works directly on the solutions (as in our example below).

Each application of an optimization algorithm to a problem instance will also be provided with a random number generator and it *must* only use this random number generator for randomization and no other sources of randomness.
You can set the seed for this random number generator via [`ex.set_rand_seed(...)`](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.execution.Execution.set_rand_seed).
If you create two identical executions and set the same seeds for both of them, the algorithms will make the same random decisions and hence should return the same results.

Furthermore, you can also set the maximum number of candidate solutions that the optimization algorithm is allowed to investigate via [`ex.set_max_fes(...)`](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.execution.Execution.set_max_fes), the maximum runtime budget in milliseconds via [`ex.set_max_time_millis(...)`](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.execution.Execution.set_max_time_millis), and a goal objective value via [`ex.set_goal_f(...)`](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.execution.Execution.set_goal_f) (the algorithm should stop after reaching it).
Notice that optimization algorithms may not terminate unless the system tells them so, so you should always specify at least either a maximum number of objective function evaluations or a runtime limit.
If you only specify a goal objective value and the algorithm cannot reach it, it may not terminate.

Finally, you can also set the path to a log file via [`ex.set_log_file(...)`](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.execution.Execution.set_log_file).
If you specify a log file, the system will automatically gather system information, collect the progress of the algorithm (in terms of improving moves by default), and the final results.
It will store all of that in a text file *after* the algorithm has completed and you have left the process scope (see below). 

Anyway, after you have completed building the execution, you can run the process you have configured via `ex.execute()`.
This method returns an instance of [`Process`](https://thomasweise.github.io/moptipy/moptipy.api.html#module-moptipy.api.process).
From the algorithm perspective, this instance provides all the information and tools that is needed to create, copy, and evaluate solutions, as well as the termination criterion that tells it when to stop.
For us, the algorithm user, it provides the information about the end result, the consumed FEs, and the end result quality.
In the code below, we illustrate how to extract these information.
Notice that you *must* always use the instances of `Process` in a [`with` block](https://peps.python.org/pep-0343/):
Once this block is left, the log file will be written.
If you use it outside of a `with` block, no log file will be generated.

Let us now look at a concrete example, which is also available as file [examples/single_run_ea1plus1_onemax](./examples/single_run_ea1plus1_onemax.html).
As example domain, we use [bit strings](https://thomasweise.github.io/moptipy/moptipy.spaces.html#module-moptipy.spaces.bitstrings) of length `n = 10` and try to solve the well-known [`OneMax`](https://thomasweise.github.io/moptipy/moptipy.examples.bitstrings.html#module-moptipy.examples.bitstrings.onemax) problem using the well-known [`(1+1) EA`](https://thomasweise.github.io/moptipy/moptipy.algorithms.html#module-moptipy.algorithms.ea1plus1).

```python
from moptipy.algorithms.ea1plus1 import EA1plus1
from moptipy.api.execution import Execution
from moptipy.examples.bitstrings.onemax import OneMax
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.operators.bitstrings.op1_m_over_n_flip import Op1MoverNflip
from moptipy.spaces.bitstrings import BitStrings
from moptipy.utils.temp import TempFile

n = 10  # we chose dimension 10
space = BitStrings(n)  # search in bit strings of length 10
problem = OneMax(n)  # we maximize the number of 1 bits
algorithm = EA1plus1(  # create (1+1)-EA that
  Op0Random(),  # starts with a random bit string and
  Op1MoverNflip(n=n, m=1))  # flips each bit with probability=1/n

# We execute the whole experiment in a temp directory.
# For a real experiment, you would put an existing directory path in `td`
# by doing `from moptipy.utils.path import Path; td = Path.directory("mydir")`
# and not use the `with` block.
with TempFile.create() as tf:  # create temporary file `tf`
  ex = Execution()  # begin configuring execution
  ex.set_solution_space(space)  # set solution space
  ex.set_objective(problem)  # set objective function
  ex.set_algorithm(algorithm)  # set algorithm
  ex.set_rand_seed(199)  # set random seed to 199
  ex.set_log_file(tf)  # set log file = temp file `tf`
  ex.set_max_fes(100)  # allow at most 100 function evaluations
  with ex.execute() as process:  # now run the algorithm*problem combination
    end_result = process.create()  # create empty record to receive result
    process.get_copy_of_best_y(end_result)  # obtain end result
    print(f"Best solution found: {process.to_str(end_result)}")
    print(f"Quality of best solution: {process.get_best_f()}")
    print(f"Consumed Runtime: {process.get_consumed_time_millis()}ms")
    print(f"Total FEs: {process.get_consumed_fes()}")

  print("\nNow reading and printing all the logged data:")
  print(tf.read_all_str())  # instead, we load and print the log file
# The temp file is deleted as soon as we leave the `with` block.
```

The output we would get from this program could look something like this:

```
Best solution found: TTTTTTTTTT
Quality of best solution: 0
Consumed Runtime: 114ms
Total FEs: 31

Now reading and printing all the logged data:
BEGIN_STATE
totalFEs: 31
totalTimeMillis: 115
bestF: 0
lastImprovementFE: 31
lastImprovementTimeMillis: 114
END_STATE
BEGIN_SETUP
p.name: ProcessWithoutSearchSpace
p.class: moptipy.api._process_no_ss._ProcessNoSS
p.maxFEs: 100
p.goalF: 0
p.randSeed: 199
...
END_SETUP
BEGIN_SYS_INFO
...
END_SYS_INFO
BEGIN_RESULT_Y
TTTTTTTTTT
END_RESULT_Y
```

You can also compare this output to the [example for log files](#413-example) further down this text.


### 3.2. How to Run a Series of Experiments

When we develop algorithms or do research, then we cannot just apply an algorithm once to a problem instance and call it a day.
Instead, we will apply multiple algorithms (or algorithm setups) to multiple problem instances and execute several runs for each algorithm * instance combination.
Our system of course also provides the facilities for this.

The concept for this is rather simple.
We distinguish "instances" and "setups."
An "instance" can be anything that a represents one specific problem instance.
It could be a string with its identifying name, it could be the [objective function](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.objective.Objective) itself, or a data structure with the instance data (as is the case for the Job Shop Scheduling Problem used in our book, where we use the class [Instance](https://thomasweise.github.io/moptipy/moptipy.examples.jssp.html#module-moptipy.examples.jssp.instance)).
The important thing is that the `__str__` method of the instance object will return a short string that can be used in file names of [log files](#41-log-files).

The second concept to understand here are "setups."
A "setup" is basically an almost fully configured [`Execution`](https://thomasweise.github.io/moptipy/moptipy.api.html#module-moptipy.api.execution) (see the [previous section](#31-how-to-apply-1-optimization-algorithm-once-to-1-problem-instance) for a detailed discussion of Executions.)
The only things that need to be left blank are the log file path and random seed, which will be filled automatically by our system.

You will basically provide a sequence of callables, i.e., functions or lambdas, each of which will return one "instance."
Additionally, you provide a sequence of callables (functions or lambdas), each of which receiving one "instance" as input and should return an almost fully configured [`Execution`](https://thomasweise.github.io/moptipy/moptipy.api.html#module-moptipy.api.execution).
You also provide the number of runs to be executed per "setup" * "instance" combination and a base directory path identifying the directory where one log file should be written for each run.
Additionally, you can specify the number of parallel processes to use, unless you want the system to automatically decide this.
All of this is passed to the function `run_experiment` in module [`moptipy.api.experiment`](https://thomasweise.github.io/moptipy/moptipy.api.html#module-moptipy.api.experiment).

This function will do all the work and generate a [folder structure](#411-file-names-and-folder-structure) of log files.
It will spawn the right number of processes, use your functions to generate "instances" and "setups," execute and them.
It will also automatically determine the random seed for each run.
The random seed sequence per instance will be the same for algorithm setups, which means that different algorithms would still start with the same solutions if they sample the first solution in the same way.
The system will even do "warmup" runs, i.e., very short dummy runs with the algorithms that are just used to make sure that the interpreter has seen all code before actually doing the experiments to avoid strange timing problems.

Below, we show one example for the automated experiment execution facility, which applies two algorithms to four problem instances with five runs per setup.
We use again the  [bit strings domain](https://thomasweise.github.io/moptipy/moptipy.spaces.html#module-moptipy.spaces.bitstrings).
We explore two problems ([`OneMax`](https://thomasweise.github.io/moptipy/moptipy.examples.bitstrings.html#module-moptipy.examples.bitstrings.onemax) and [`LeadingOnes`](https://thomasweise.github.io/moptipy/moptipy.examples.bitstrings.html#module-moptipy.examples.bitstrings.leadingones)) of two different sizes each, leading to four problem instances in total.
We apply the well-known [`(1+1) EA`](https://thomasweise.github.io/moptipy/moptipy.algorithms.html#module-moptipy.algorithms.ea1plus1) as well as the trivial [random sampling](https://thomasweise.github.io/moptipy/moptipy.algorithms.html#module-moptipy.algorithms.random_sampling).

The code below is available as file [examples/experiment_2_algorithms_4_problems](./examples/experiment_2_algorithms_4_problems.html).
Besides executing the experiment, it also prints the end results obtained from parsing the log files (see [Section 4.2.](#42-end-result-csv-files) for more information). 

```python
from moptipy.algorithms.ea1plus1 import EA1plus1
from moptipy.algorithms.random_sampling import RandomSampling
from moptipy.api.execution import Execution
from moptipy.api.experiment import run_experiment
from moptipy.evaluation.end_results import EndResult
from moptipy.examples.bitstrings.leadingones import LeadingOnes
from moptipy.examples.bitstrings.onemax import OneMax
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.operators.bitstrings.op1_m_over_n_flip import Op1MoverNflip
from moptipy.spaces.bitstrings import BitStrings
from moptipy.utils.temp import TempDir

# The four problems we want to try to solve:
problems = [lambda: OneMax(10),  # 10-dimensional OneMax
            lambda: OneMax(32),  # 32-dimensional OneMax
            lambda: LeadingOnes(10),  # 10-dimensional LeadingOnes
            lambda: LeadingOnes(32)]  # 32-dimensional LeadingOnes


def make_ea1plus1(problem) -> Execution:
    """
    Create a (1+1)-EA Execution.

    :param problem: the problem (OneMax or LeadingOnes)
    :returns: the execution
    """
    ex = Execution()
    ex.set_solution_space(BitStrings(problem.n))
    ex.set_objective(problem)
    ex.set_algorithm(
        EA1plus1(  # create (1+1)-EA that
            Op0Random(),  # starts with a random bit string and
            Op1MoverNflip(n=problem.n, m=1)))  # flips each bit with p=1/n
    ex.set_max_fes(100)  # permit 100 FEs
    return ex


def make_random_sampling(problem) -> Execution:
    """
    Create a Random Sampling Execution.

    :param problem: the problem (OneMax or LeadingOnes)
    :returns: the execution
    """
    ex = Execution()
    ex.set_solution_space(BitStrings(problem.n))
    ex.set_objective(problem)
    ex.set_algorithm(RandomSampling(Op0Random()))
    ex.set_max_fes(100)
    return ex


# We execute the whole experiment in a temp directory.
# For a real experiment, you would put an existing directory path in `td`
# by doing `from moptipy.utils.path import Path; td = Path.directory("mydir")`
# and not use the `with` block.
with TempDir.create() as td:  # create temporary directory `td`
    run_experiment(base_dir=td,  # set the base directory for log files
                   instances=problems,  # define the problem instances
                   setups=[make_ea1plus1,  # provide (1+1)-EA run creator
                           make_random_sampling],  # provide RS run creator
                   n_runs=5,  # we will execute 5 runs per setup
                   n_threads=1)  # we use only a single thread here

    EndResult.from_logs(  # parse all log files and print end results
        td, lambda er: print(f"{er.algorithm} on {er.instance}: {er.best_f}"))
# The temp directory is deleted as soon as we leave the `with` block.
```

The output of this program, minus the status information, could look roughly like this:

```
ea1p1_1_over_n_flip_T on onemax_10: 0
ea1p1_1_over_n_flip_T on onemax_10: 0
ea1p1_1_over_n_flip_T on onemax_10: 0
ea1p1_1_over_n_flip_T on onemax_10: 0
ea1p1_1_over_n_flip_T on onemax_10: 0
ea1p1_1_over_n_flip_T on onemax_32: 3
ea1p1_1_over_n_flip_T on onemax_32: 2
ea1p1_1_over_n_flip_T on onemax_32: 4
ea1p1_1_over_n_flip_T on onemax_32: 1
ea1p1_1_over_n_flip_T on onemax_32: 0
ea1p1_1_over_n_flip_T on leadingones_32: 22
ea1p1_1_over_n_flip_T on leadingones_32: 20
ea1p1_1_over_n_flip_T on leadingones_32: 19
ea1p1_1_over_n_flip_T on leadingones_32: 18
ea1p1_1_over_n_flip_T on leadingones_32: 28
ea1p1_1_over_n_flip_T on leadingones_10: 0
ea1p1_1_over_n_flip_T on leadingones_10: 1
ea1p1_1_over_n_flip_T on leadingones_10: 1
ea1p1_1_over_n_flip_T on leadingones_10: 0
ea1p1_1_over_n_flip_T on leadingones_10: 0
rs on onemax_10: 0
rs on onemax_10: 2
rs on onemax_10: 1
rs on onemax_10: 2
rs on onemax_10: 1
rs on onemax_32: 8
rs on onemax_32: 8
rs on onemax_32: 8
rs on onemax_32: 9
rs on onemax_32: 9
rs on leadingones_32: 26
rs on leadingones_32: 26
rs on leadingones_32: 25
rs on leadingones_32: 26
rs on leadingones_32: 23
rs on leadingones_10: 4
rs on leadingones_10: 0
rs on leadingones_10: 3
rs on leadingones_10: 3
rs on leadingones_10: 0
```


### 3.3. How to Solve an Optimization Problem

If you want to solve an optimization problem with [moptipy](https://thomasweise.github.io/moptipy), then you need at least the following three things:

1. a space `Y` of possible solutions,
2. an objective function `f`  rating the solutions, i.e., which maps elements `y` of `Y` to either integer or float numbers, where *smaller* values are better, and
3. an optimization algorithm that navigates through `Y` and tries to find solutions `y` in `Y` with low corresponding values `f(y)`.

You may need more components, but if you have these three, then you can [run an experiment](#32-how-to-run-a-series-of-experiments).


#### 3.3.1. Define a New Problem Type

At the core of all optimization problems lies the objective function.
All objective functions in `moptipy` are instances of the class [Objective](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.objective.Objective).
If you want to add a new optimization problem, you must derive a new subclass from this class. 

There are two functions you must implement:

- [`evaluate(x)`](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.objective.Objective.evaluate) receives a candidate solution `x` as input and must return either an `int` or a `float` rating its quality (smaller values are *better*) and
- `__str__()` returns a string representation of the objective function and may be used in file names and folder structures (depending on how you execute your experiments).
  It therefore must not contain spaces and other dodgy characters.

Additionally, you *may* implement the following two functions

- [`lower_bound()`](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.objective.Objective.lower_bound) returns either an `int` or a `float` with the lower bound of the objective value.
  This value does not need to be an objective value that can actually be reached, but if you implement this function, then the value must be small enough so that it is *impossible* to ever reach a smaller objective value.
  If we execute an experiment and no goal objective value is specified, then the system will automatically use this lower bound if it is present.
  Then, if any solution `x` with `f.evaluate(x)==f.lower_bound()` is encountered, the optimization process is automatically stopped.
  Furthermore, after the optimization process is stopped, it is verified that the final solution does not have an objective value smaller than the lower bound.
  If it does, then we throw an exception.
- [`upper_bound()`](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.objective.Objective.upper_bound) returns either an `int` or a `float` with the upper bound of the objective value.
  This value does not need to be an objective value that can actually be reached, but if you implement this function, then the value must be large enough so that it is *impossible* to ever reach a larger objective value.
  This function, if present, is used to validate the objective value of the final result of the optimization process.

OK, with this information we are basically able to implement our own problem.
Here, we define the task "sort n numbers" as optimization problem.
Basically, we want that our optimization algorithm works on [permutations](https://thomasweise.github.io/moptipy/moptipy.spaces.html#module-moptipy.spaces.permutations) of `n` numbers and is searching for the sorted permutation.
As objective value, we count the number of "sorting errors" in a permutation.
If the number at index `i` is bigger than the number at index `i+1`, then this is a sorting error.
If `n=5`, then the permutation `0,1,2,3,4` has no sorting error, i.e., the best possible objective value `0`.
The permutation `4,3,2,1,0` has `n-1=4` sorting errors, i.e., is the worst possible solution.
The permutation `3,4,2,0,1` as `2` sorting errors.

From these thoughts, we also know that we can implement `lower_bound()` to return 0 and `upper_bound()` to return `n-1`.
`__str__` could be `"sort" + n`, i.e., `sort5` in the above example where `n=5`.

We provide the corresponding code in [Section 3.3.3](#333-applying-an-own-algorithm-to-an-own-problem) below.


#### 3.3.2. Define a New Algorithm

While `moptipy` comes with several well-known algorithms out-of-the-box, you can of course also implement your own [algorithms](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.algorithm.Algorithm).
These can then make use of the existing [spaces](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.space.Space) and [search operators](https://thomasweise.github.io/moptipy/moptipy.api.html#module-moptipy.api.operators) &ndash; or not.
Let us here create an example algorithm implementation that does *not* use any of the pre-defined [search operators](https://thomasweise.github.io/moptipy/moptipy.api.html#module-moptipy.api.operators).

All optimization algorithms must be subclasses of the class [Algorithm](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.algorithm.Algorithm).
Each of them must implement two methods:

- [`solve(process)`](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.algorithm.Algorithm.solve) receives an instance of [`Process`](https://thomasweise.github.io/moptipy/moptipy.api.html#module-moptipy.api.process), which provides the operations to work with the search space, to evaluate solutions, the termination criterion, and the random number generator.
- `__str__()` must return a short string representation identifying the algorithm and its setup.
  This string will be used in file and folder names and therefore must not contain spaces or otherwise dodgy characters.

The instance `process` of [`Process`](https://thomasweise.github.io/moptipy/moptipy.api.html#module-moptipy.api.process) passed to the function [`solve`](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.algorithm.Algorithm.solve) is a key element of our `moptipy` API.
If the algorithm needs a data structure to hold a point in the search space, it should invoke [`process.create()`](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.space.Space.create).
If it needs to copy the point `source` to the point `dest`, it should invoke [`process.copy(dest, source)`](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.space.Space.copy).

If it wants to know the quality of the point `x`, it should invoke [`process.evaluate(x)`](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.process.Process.evaluate).
This function will actually forward the call to the actual objective function (see, e.g., [Section 3.3.1](#331-define-a-new-problem-type) above).
However, it will do more:
It will automatically keep track of the best-so-far solution and, if needed, build logging information in memory.

Before every single call to [`process.evaluate()`](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.process.Process.evaluate), you should invoke [`process.should_terminate()`](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.process.Process.should_terminate).
This function returns `True` if the optimization algorithm should stop whatever it is doing and return.
This can happen when a solution of sufficiently good quality is reached, when the maximum number of FEs is exhausted, or when the computational budget in terms of runtime is exhausted.

Since many optimization algorithms make random choices, the function [`process.get_random()`](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.process.Process.get_random) returns a [random number generator](https://numpy.org/doc/stable/reference/random/generator.html).
This generator *must* be the only source of randomness used by an algorithm.
It will automatically be seeded by our system, allowing for repeatable and reproducible runs.

The `process` also can provide information about the best-so-far solution, the consumed runtime and FEs, as well as when the last improvement was achieved.
Anyway, all interaction between the algorithm and the actual optimization algorithm will happen through the `process` object.

Equipped with this information, we can develop a simple and rather stupid algorithm to attack the sorting problem.
The search space that we use are the [permutations](https://thomasweise.github.io/moptipy/moptipy.spaces.html#module-moptipy.spaces.permutations) of `n` numbers.
(These will be internally represented as [numpy `ndarray`s](https://numpy.org/doc/stable/reference/arrays.ndarray.html), but we do not need to bother with this, as we this is done automatically for us.) 
Our algorithm should start with allocating a point `x_cur` in the search space, filling it with the numbers `0..n-1`, and shuffling it randomly.
For the shuffling, it will use than random number generator provided by `process`.
It will evaluate this solution and remember its quality in variable `f_cur`.
It will also allocate a second container `x_new` for permutations.

In each step, our algorithm will copy `x_cur` to `x_new`.
Then, it will use the random number generator to draw two numbers `i` and `j` from `0..n-1`.
It will swap the two numbers at these indices in `x_new`, i.e., exchange `x_new[i], x_new[j] = x_new[j], x_new[i]`.
We then evaluate `x_new` and if the resulting objective value `f_new` is better than `f_cur`, we swap `x_new` and `x_cur` (which is faster than copying `x_new` to `x_cur`) and store `f_new` in `f_cur`.
We repeat this until [`process.should_terminate()`](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.process.Process.should_terminate) becomes `True`.
All of this is implemented in the source code example below in [Section 3.3.3](#333-applying-an-own-algorithm-to-an-own-problem).


#### 3.3.3. Applying an Own Algorithm to an Own Problem

The following code combines our [own algorithm](#332-define-a-new-algorithm) and our [own problem type](#331-define-a-new-problem-type) that we discussed in the prior two sections and executes an experiment.
It is available as file [examples/experiment_own_algorithm_and_problem](./examples/experiment_own_algorithm_and_problem.html).
Notice how we provide functions for generating both the problem instances (here the objective functions) and the algorithm setups exactly as we described in [Section 3.2.](#32-how-to-run-a-series-of-experiments) above.

```python
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

        :returns: 0
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
# For a real experiment, you would put an existing directory path in `td`
# by doing `from moptipy.utils.path import Path; td = Path.directory("mydir")`
# and not use the `with` block.
with TempDir.create() as td:  # create temporary directory `td`
    run_experiment(base_dir=td,  # set the base directory for log files
                   instances=problems,  # define the problem instances
                   setups=[make_execution],  # creator for our algorithm
                   n_runs=5,  # we will execute 5 runs per setup
                   n_threads=1)  # we use only a single thread here

    EndResult.from_logs(  # parse all log files and print end results
        td, lambda er: print(f"{er.algorithm} on {er.instance}: {er.best_f}"))
# The temp directory is deleted as soon as we leave the `with` block.
```

The output of this program, minus status output, could look like this:

```
myAlgo on sort10: 2
myAlgo on sort10: 2
myAlgo on sort10: 1
myAlgo on sort10: 1
myAlgo on sort10: 2
myAlgo on sort100: 35
myAlgo on sort100: 41
myAlgo on sort100: 33
myAlgo on sort100: 34
myAlgo on sort100: 35
myAlgo on sort5: 1
myAlgo on sort5: 1
myAlgo on sort5: 1
myAlgo on sort5: 1
myAlgo on sort5: 1
```

## 4. Data Formats

We develop several data formats to store and evaluate the results of computational experiments with our `moptipy` software.
Here you can find their basic definitions.


### 4.1. Log Files

#### 4.1.1. File Names and Folder Structure

One independent run of an algorithm on one problem instance produces one log file.
Each run is identified by the algorithm that is applied, the problem instance to which it is applied, and the random seed.
This tuple is reflected in the file name.
`ea1p1_swap2_demo_0x5a9363100a272f12.txt`, for example, represents the algorithm `ea1p1_swap2` applied to the problem instance `demo` and started with random seed `0x5a9363100a272f12` (where `0x` stands for hexademical notation).
The log files are grouped in a `algorithm`-`instance` folder structure.
In the above example, there would be a folder `ea1p1_swap2` containing a folder `demo`, which, in turn, contains all the log files from all runs of that algorithm on this instance.


#### 4.1.2. Log File Sections

A log file is a simple text file divided into several sections.
Each section `X` begins with the line `BEGIN_X` and ends with the line `END_X`.
There are three types of sections:

- *[Semicolon-separated values](https://thomasweise.github.io/moptipy/moptipy.utils.html#moptipy.utils.logger.CsvSection)* can hold a series of data values, where each row is divided into multiple values and the values are separated by `;`
- *[Key-values](https://thomasweise.github.io/moptipy/moptipy.utils.html#moptipy.utils.logger.KeyValueSection)* sections represent, well, values for keys in form of a mapping compatible with [YAML](https://yaml.org/spec/1.2/spec.html#mapping).
  In other words, each line contains a key, followed by `: `, followed by the value.
  The keys can be hierarchically structured in scopes, for example `a.b` and `a.c` indicate two keys `b` and `c` that belong to scope `a`.
- *[Raw text](https://thomasweise.github.io/moptipy/moptipy.utils.html#moptipy.utils.logger.TextSection)* sections contain text without a general or a priori structure, e.g., the string representation of the best solutions found.

In all the above sections, all the character `#` is removed from output.
The character `#` indicates a starting comment and can only be written by the routines dedicated to produce comments.


##### The Section `PROGRESS`

When setting up an algorithm execution, you can specify whether you want to [log the progress](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.execution.Execution.set_log_improvements) of the algorithm (or not).
If and only if you choose to log the progress, the `PROGRESS` section will be contained in the log file.
You can also choose if you want to [log all algorithm steps](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.execution.Execution.set_log_all_fes) or [only the improving moves](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.execution.Execution.set_log_improvements), the latter being the default behavior.
Notice that even if you do not choose to log the algorithm's progress, the section `STATE` with the objective value of the best solution encountered, the FE when it was found, and the consumed runtime as well as the `RESULT_*` sections with the best encountered candidate solution and point in the search space will be included.

The `PROGRESS` section contains log points describing the algorithm progress over time in a semicolon-separated values format with one data point per line.
It has an internal header describing the data columns.
There will at least be the following columns:

1. `fes` denoting the integer number of performed objective value evaluations
2. `timeMS` the clock time that has passed since the start of the run, measured in milliseconds and stored as integer value.
   Python actually provides the system clock time in terms of nanoseconds, however, we always round up to the next highest millisecond.
   We believe that milliseconds are a more reasonable time measure here and a higher resolution is probably not helpful anyway.
3. `f` the best-so-far objective value

This configuration is denoted by the header `fes;timeMS;f`.
After this header and until `END_PROGRESS`, each line will contain one data point with values for the specified columns.
Notice that for each FE, there will be at most one data point but there might be multiple data points per millisecond.
This is especially true if we log all FEs.
Usually, we could log one data point for every improvement of the objective value, though.


##### The Section `STATE`

The end state when the run terminates is logged in the section `STATE` in key-value format.
It holds at least the following keys:

- `totalFEs` the total number of objective function evaluations performed, as integer
- `totalTimeMillis` the total number of clock time milliseconds elapsed since the begin of the run, as integer
- `bestF` the best objective function value encountered during the run
- `lastImprovementFE` the index of the last objective function evaluation where the objective value improved, as integer
- `lastImprovementTimeMillis` the time in milliseconds at which the last objective function value improvement was registered, as integer


##### The Section `SETUP`

In this key-value section, we log information about the configuration of the optimization algorithm as well as the parameters of the problem instance solved.
There are at least the following keys:

- [process](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.process.Process) wrapper parameters (scope `p`):
  - `p.name`: the name of the process wrapper, i.e., a short mnemonic describing its purpose
  - `p.class`: the python class of the process wrapper
  - `p.maxTimeMillis`: the maximum clock time in milliseconds, if specified
  - `p.maxFEs`: the maximum number of objective function evaluations (FEs), if specified
  - `p.goalF`: the goal objective value, if specified (or computed via the `lower_bound()` of the [objective function](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.objective.Objective))
  - `p.randSeed`: the random seed in decimal notation
  - `p.randSeed(hex)`: the random seed in hexadecimal notation
  - `p.randGenType`: the class of the random number generator
  - `p.randBitGenType`: the class of the bit generator used by the random number generator
- [algorithm](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.algorithm.Algorithm) parameters: scope `a`, includes algorithm `name`, `class`, etc.
- solution [space](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.space.Space) scope `y`, includes `name` and `class` of solution space
- [objective function](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.objective.Objective) information: scope `f`
- search [space](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.space.Space) information (if search space is different from solution space): scope `x`
- [encoding](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.encoding.Encoding) information (if encoding is defined): scope `g` 

If you implement an own [algorithm](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.algorithm.Algorithm), [objective function](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.objective.Objective), [space](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.space.Space), or your own [search operators](https://thomasweise.github.io/moptipy/moptipy.api.html#module-moptipy.api.operators), then you can overwrite the method [`log_parameters_to(logger)`](https://thomasweise.github.io/moptipy/moptipy.api.html#moptipy.api.component.Component.log_parameters_to).
This method will automatically be invoked when writing the log files of a run.
It should *always* start with calling the super implementation (`super().log_parameters_to(logger)`).
After that, you can store key-value pairs describing the parameterization of your component.
This way, such information can be preserved in log files.


##### The Section `SYS_INFO`

The system information section is again a key-value section.
It holds key-value pairs describing features of the machine on which the experiment was executed.
This includes information about the CPU, the operating system, the Python installation, as well as the version information of packages used by moptipy.


##### The `RESULT_*` Sections

The textual representation of the best encountered solution (whose objective value is noted as `bestF` in section `STATE`) is stored in the section `RESULT_Y`.
Since we can use many different solution spaces, this section just contains raw text.

If the search and solution space are different, the section `RESULT_X` is included.
It then holds the point in the search space corresponding to the solution presented in `RESULT_Y`.


##### The `ERROR_*` Sections

Our package has mechanisms to catch and store errors that occurred during the experiments.
Each type of error will be stored in a separate log section and each such sections may store the class of the error in form `exceptionType: error-class`, the error message in the form `exceptionValue: error-message` and the stack trace line by line after a line header `exceptionStackTrace:`.
The following exception sections are currently supported:

- If an exception is encountered during the algorithm run, it will be store in section `ERROR_IN_RUN`.
- If an exception occurred in the context of the optimization process, it will be stored in `ERROR_IN_CONTEXT`.
  This may be an error during the execution of the algorithm, or, more likely, an error in the code that accesses the process data afterwards, e.g., that processes the best solution encountered.
- If the validation of the finally returned candidate solution failed, the resulting error will be stored in section `ERROR_INVALID_Y`.
- If the validation of the finally returned point in the search space failed, the resulting error will be stored in section `ERROR_INVALID_X`.
- If an inconsistency in the time measurement is discovered, this will result in the section `ERROR_TIMING`.
  Such an error may be caused when the computer clock is adjusted during the run of an optimization algorithm.
  It will also occur if an algorithm terminates without performing even a single objective function evaluation.
- In the unlikely case that an exception occurs during the writing of the log but writing can somehow continue, this exception will be stored in section `ERROR_IN_LOG`.


#### 4.1.3. Example

You can execute the following Python code to obtain an example log file.
This code is also available in file [examples/log_file_jssp.py](./examples/log_file_jssp.html):

```python
from moptipy.algorithms.ea1plus1 import EA1plus1  # the algorithm we use
from moptipy.examples.jssp.experiment import run_experiment  # the JSSP runner
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle  # 0-ary op
from moptipy.operators.permutations.op1_swap2 import Op1Swap2  # 1-ary op
from moptipy.utils.temp import TempDir  # temp directory tool

# We work in a temporary directory, i.e., delete all generated files on exit.
# For a real experiment, you would put an existing directory path in `td`
# by doing `from moptipy.utils.path import Path; td = Path.directory("mydir")`
# and not use the `with` block.
with TempDir.create() as td:  # create temp directory
    # Execute an experiment consisting of exactly one run.
    # As example domain, we use the job shop scheduling problem (JSSP).
    run_experiment(
        base_dir=td,  # working directory = temp dir
        algorithms=[  # the set of algorithms to use: we use only 1
            lambda inst, pwr:  # an algorithm is created via a lambda
            EA1plus1(Op0Shuffle(pwr), Op1Swap2())],  # we use (1+1)-EA
        instances=("demo",),  # use the demo JSSP instance
        n_runs=1,  # perform exactly one run
        n_threads=1)  # use exactly one thread
    # The random seed is automatically generated based on the instance name.
    print(td.resolve_inside(  # so we know algorithm, instance, and seed
        "ea1p1_swap2/demo/ea1p1_swap2_demo_0x5a9363100a272f12.txt")
        .read_all_str())  # read file into string (which then gets printed)
# When leaving "while", the temp dir will be deleted
```

The example log file printed by the above code will then look as follows:

```
BEGIN_PROGRESS
fes;timeMS;f
1;1;267
5;1;235
10;1;230
20;1;227
25;1;205
40;1;200
84;2;180
END_PROGRESS
BEGIN_STATE
totalFEs: 84
totalTimeMillis: 2
bestF: 180
lastImprovementFE: 84
lastImprovementTimeMillis: 2
END_STATE
BEGIN_SETUP
p.name: LoggingProcessWithSearchSpace
p.class: moptipy.api._process_ss_log._ProcessSSLog
p.maxTimeMillis: 120000
p.goalF: 180
p.randSeed: 6526669205530947346
p.randSeed(hex): 0x5a9363100a272f12
p.randGenType: numpy.random._generator.Generator
p.randBitGenType: numpy.random._pcg64.PCG64
a.name: ea1p1_swap2
a.class: moptipy.algorithms.ea1plus1.EA1plus1
a.op0.name: shuffle
a.op0.class: moptipy.operators.permutations.op0_shuffle.Op0Shuffle
a.op1.name: swap2
a.op1.class: moptipy.operators.permutations.op1_swap2.Op1Swap2
y.name: gantt_demo
y.class: moptipy.examples.jssp.gantt_space.GanttSpace
y.shape: (5, 4, 3)
y.dtype: h
y.inst.name: demo
y.inst.class: moptipy.examples.jssp.instance.Instance
y.inst.machines: 5
y.inst.jobs: 4
y.inst.makespanLowerBound: 180
y.inst.makespanUpperBound: 482
y.inst.dtype: b
f.name: makespan
f.class: moptipy.examples.jssp.makespan.Makespan
x.name: perm4w5r
x.class: moptipy.spaces.permutations.Permutations
x.nvars: 20
x.dtype: b
x.min: 0
x.max: 3
x.repetitions: 5
g.name: operation_based_encoding
g.class: moptipy.examples.jssp.ob_encoding.OperationBasedEncoding
g.dtypeMachineIdx: b
g.dtypeJobIdx: b
g.dtypeJobTime: h
END_SETUP
BEGIN_SYS_INFO
session.start: 2022-03-18 14:43:51.707977
session.node: home
session.procesId: 0x82a68
session.cpuAffinity: 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
session.ipAddress: 127.0.0.1
version.moptipy: 0.8.1
version.numpy: 1.21.5
version.numba: 0.55.1
version.matplotlib: 3.5.1
version.psutil: 5.9.0
version.scikitlearn: 1.0.2
hardware.machine: x86_64
hardware.nPhysicalCpus: 8
hardware.nLogicalCpus: 16
hardware.cpuMhz: (2200MHz..3700MHz)*16
hardware.byteOrder: little
hardware.cpu: AMD Ryzen 7 2700X Eight-Core Processor
hardware.memSize: 33605500928
python.version: 3.9.7 (default, Sep 10 2021, 14:59:43) [GCC 11.2.0]
python.implementation: CPython
os.name: Linux
os.release: 5.13.0-35-generic
os.version: 40-Ubuntu SMP Mon Mar 7 08:03:10 UTC 2022
END_SYS_INFO
BEGIN_RESULT_Y
1,20,30,0,30,40,3,145,165,2,170,180,1,0,20,0,40,60,2,60,80,3,165,180,2,0,30,0,60,80,1,80,130,3,130,145,1,30,60,3,60,90,0,90,130,2,130,170,3,0,50,2,80,92,1,130,160,0,160,170
END_RESULT_Y
BEGIN_RESULT_X
2,1,3,1,0,0,2,0,1,2,3,1,0,2,1,3,0,3,2,3
END_RESULT_X
```


### 4.2. End Result CSV Files

While a [log file](#41-log-files) contains all the data of a single run, you often want to get just the basic measurements, such as the result objective values, from all runs of one experiment in a single file.
The class [`moptipy.evaluation.end_results.EndResult`](https://thomasweise.github.io/moptipy/moptipy.evaluation.html#moptipy.evaluation.end_results.EndResult) provides the tools needed to parse all log files, extract these information, and store them into a semicolon-separated-values formatted file.
The files generated this way can easily be imported into applications like Microsoft Excel.

#### 4.2.1. The End Results File Format

An end results file contains a header line and then one line for each log file that was parsed.
The eleven columns are separated by `;`.
Cells without value are left empty.

It presents the following columns:

1. `algorithm`: the algorithm that was executed
2. `instance`: the instance it was applied to
3. `randSeed` the hexadecimal version of the random seed of the run
4. `bestF`: the best objective value encountered during the run
5. `lastImprovementFE`: the FE when the last improvement was registered
6. `lastImprovementTimeMillis`: the time in milliseconds from the start of the run when the last improvement was registered
7. `totalFEs`: the total number of FEs performed
8. `totalTimeMillis`: the total time in milliseconds consumed by the run
9. `goalF`: the goal objective value, if specified (otherwise empty)
10. `maxFEs`: the computational budget in terms of the maximum number of permitted FEs, if specified (otherwise empty)
11. `maxTimeMillis`: the computational budget in terms of the maximum runtime in milliseconds, if specified (otherwise empty)

For each run, i.e., algorithm x instance x seed combination, one row with the above values is generated.
Notice that from the algorithm and instance name together with the random seed, you can find the corresponding log file.


#### 4.2.2. An Example for End Results Files

Let us execute an abridged example experiment, parse all log files, condense their information into an end results statistics file, and then print that file's contents.
We can do that with the code below, which is also available as file [examples/end_results_jssp.py](./examples/end_results_jssp.html).

```python
from moptipy.algorithms.ea1plus1 import EA1plus1  # first algo to test
from moptipy.algorithms.hill_climber import HillClimber  # second algo to test
from moptipy.evaluation.end_results import EndResult  # the end result record
from moptipy.examples.jssp.experiment import run_experiment  # JSSP example
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle  # 0-ary op
from moptipy.operators.permutations.op1_swap2 import Op1Swap2  # 1-ary op
from moptipy.utils.temp import TempDir  # tool for temp directories

# We work in a temporary directory, i.e., delete all generated files on exit.
# For a real experiment, you would put an existing directory path in `td`
# by doing `from moptipy.utils.path import Path; td = Path.directory("mydir")`
# and not use the `with` block.
with TempDir.create() as td:
    run_experiment(  # run the JSSP experiment with the following parameters:
        base_dir=td,  # base directory to write all log files to
        algorithms=[  # the set of algorithm generators
            lambda inst, pwr: EA1plus1(Op0Shuffle(pwr), Op1Swap2()),  # algo 1
            lambda inst, pwr: HillClimber(Op0Shuffle(pwr), Op1Swap2())],  # 2
        instances=("demo", "abz7", "la24"),  # we use 3 JSSP instances
        max_fes=10000,  # we grant 10000 FEs per run
        n_runs=4,  # perform 4 runs per algorithm * instance combination
        n_threads=1)  # we use only a single thread here

    end_results = []  # this list will receive the end results records
    EndResult.from_logs(td, end_results.append)  # get results from log files

    er_csv = EndResult.to_csv(  # store end results to csv file (returns path)
        end_results,  # the list of end results to store
        td.resolve_inside("end_results.txt"))  # path to the file to generate
    print(er_csv.read_all_str())  # read generated file as string and print it
# When leaving "while", the temp dir will be deleted
```

This will yield something like the following output:

```
algorithm;instance;randSeed;bestF;lastImprovementFE;lastImprovementTimeMillis;totalFEs;totalTimeMillis;goalF;maxFEs;maxTimeMillis
hc_swap2;la24;0xac5ca7763bbe7138;1233;2349;42;10000;187;935;10000;120000
hc_swap2;la24;0x23098fe72e435030;1065;9868;167;10000;185;935;10000;120000
hc_swap2;la24;0xb76a45e4f8b431ae;1118;2130;34;10000;138;935;10000;120000
hc_swap2;la24;0xb4eab9a0c2193a9e;1111;2594;46;10000;148;935;10000;120000
hc_swap2;abz7;0x3e96d853a69f369d;826;8335;111;10000;160;656;10000;120000
hc_swap2;abz7;0x7e986b616543ff9b;850;6788;100;10000;189;656;10000;120000
hc_swap2;abz7;0xeb6420da7243abbe;804;3798;67;10000;215;656;10000;120000
hc_swap2;abz7;0xd3de359d5e3982fd;814;4437;60;10000;161;656;10000;120000
hc_swap2;demo;0xdac201e7da6b455c;205;4;1;10000;132;180;10000;120000
hc_swap2;demo;0x5a9363100a272f12;200;33;1;10000;127;180;10000;120000
hc_swap2;demo;0x9ba8fd0486c59354;180;34;1;34;2;180;10000;120000
hc_swap2;demo;0xd2866f0630434df;185;128;3;10000;139;180;10000;120000
ea1p1_swap2;la24;0xb4eab9a0c2193a9e;1033;7503;127;10000;185;935;10000;120000
ea1p1_swap2;la24;0x23098fe72e435030;1026;9114;157;10000;188;935;10000;120000
ea1p1_swap2;la24;0xac5ca7763bbe7138;1015;9451;123;10000;140;935;10000;120000
ea1p1_swap2;la24;0xb76a45e4f8b431ae;1031;5218;89;10000;185;935;10000;120000
ea1p1_swap2;abz7;0x7e986b616543ff9b;767;9935;153;10000;183;656;10000;120000
ea1p1_swap2;abz7;0xeb6420da7243abbe;756;8005;109;10000;164;656;10000;120000
ea1p1_swap2;abz7;0xd3de359d5e3982fd;762;9128;129;10000;173;656;10000;120000
ea1p1_swap2;abz7;0x3e96d853a69f369d;761;9663;134;10000;167;656;10000;120000
ea1p1_swap2;demo;0xdac201e7da6b455c;180;83;2;83;3;180;10000;120000
ea1p1_swap2;demo;0x5a9363100a272f12;180;84;2;84;3;180;10000;120000
ea1p1_swap2;demo;0x9ba8fd0486c59354;180;33;1;33;2;180;10000;120000
ea1p1_swap2;demo;0xd2866f0630434df;180;63;2;63;3;180;10000;120000
```

### 4.3. End Result Statistics CSV Files

We can also aggregate the end result data over either algorithm x instance combinations, over whole algorithms, over whole instances, or just over everything.
The class [`moptipy.evaluation.end_statistics.EndStatistics`](https://thomasweise.github.io/moptipy/moptipy.evaluation.html#moptipy.evaluation.end_statistics.EndStatistics) provides the tools needed to aggregate statistics over sequences of [`moptipy.evaluation.end_results.EndResult`](https://thomasweise.github.io/moptipy/moptipy.evaluation.html#moptipy.evaluation.end_results.EndResult) and to store them into a semicolon-separated-values formatted file.
The files generated this way can easily be imported into applications like Microsoft Excel.

#### 4.3.1. The End Result Statistics File Format

End result statistics files contain information in form of statistics aggregated over several runs.
Therefore, they first contain columns identifying the data over which has been aggregated:

1. `algorithm`: the algorithm used (empty if we aggregate over all algorithms)
2. `instance`: the instance to which it was applied (empty if we aggregate over all instance)

Then the column `n` denotes the number of runs that were performed in the above setting. 
We have then the following data columns:

1. `bestF.x`: the best objective value encountered during the run
2. `lastImprovementFE.x`: the FE when the last improvement was registered
3. `lastImprovementTimeMillis.x`: the time in milliseconds from the start of the run when the last improvement was registered
4. `totalFEs.x`: the total number of FEs performed
5. `totalTimeMillis.x`: the total time in milliseconds consumed by the run

Here, the `.x` can stand for the following statistics:

- `min`: the minimum
- `med`: the median
- `mean`: the mean
- `geom`: the geometric mean
- `max`: the maximum
- `sd`: the standard deviation

The column `goalF` denotes the goal objective value, if any.
If it is not empty, then we also have the columns `bestFscaled.x`, which provide statistics of `bestF/goalF` as discussed above.
If `goalF` is defined for at least some settings, we also get the following columns:

1. `nSuccesses`: the number of runs that were successful in reaching the goal
2. `successFEs.x`: the statistics about the FEs until success, but *only* computed over the successful runs
3. `successTimeMillis.x`: the statistics of the runtime until success, but *only* computed over the successful runs
4. `ertFEs`: the empirically estimated runtime to success in FEs
5. `ertTimeMillis`: the empirically estimated runtime to success in milliseconds

Finally, the columns `maxFEs` and `maxTimeMillis`, if specified, include the computational budget limits in terms of FEs or milliseconds.


#### 4.3.2. Example for End Result Statistics Files

We can basically execute the same abridged experiment as in the [previous section](#422-an-example-for-end-results-files), but now take the aggregation of information one step further with the code below.
This code is also available as file [examples/end_statistics_jssp](./examples/end_statistics_jssp.html).

```python
from moptipy.algorithms.ea1plus1 import EA1plus1  # first algo to test
from moptipy.algorithms.hill_climber import HillClimber  # second algo to test
from moptipy.evaluation.end_results import EndResult  # the end result record
from moptipy.evaluation.end_statistics import EndStatistics  # statistics rec
from moptipy.examples.jssp.experiment import run_experiment  # JSSP example
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle  # 0-ary op
from moptipy.operators.permutations.op1_swap2 import Op1Swap2  # 1-ary op
from moptipy.utils.temp import TempDir  # tool for temp directories

# We work in a temporary directory, i.e., delete all generated files on exit.
# For a real experiment, you would put an existing directory path in `td`
# by doing `from moptipy.utils.path import Path; td = Path.directory("mydir")`
# and not use the `with` block.
with TempDir.create() as td:
    run_experiment(  # run the JSSP experiment with the following parameters:
        base_dir=td,  # base directory to write all log files to
        algorithms=[  # the set of algorithm generators
            lambda inst, pwr: EA1plus1(Op0Shuffle(pwr), Op1Swap2()),  # algo 1
            lambda inst, pwr: HillClimber(Op0Shuffle(pwr), Op1Swap2())],  # 2
        instances=("demo", "abz7", "la24"),  # we use 3 JSSP instances
        max_fes=10000,  # we grant 10000 FEs per run
        n_runs=4,  # perform 4 runs per algorithm * instance combination
        n_threads=1)  # we use only a single thread here

    end_results = []  # this list will receive the end results records
    EndResult.from_logs(td, end_results.append)  # get results from log files

    end_stats = []  # the list to receive the statistics records
    EndStatistics.from_end_results(  # compute the end result statistics for
        end_results, end_stats.append)  # each algorithm*instance combination

    es_csv = EndStatistics.to_csv(  # store the statistics to a CSV file
        end_stats, td.resolve_inside("end_stats.txt"))
    print(es_csv.read_all_str())  # read and print the file
# When leaving "while", the temp dir will be deleted
```

We will get the following output:

```
algorithm;instance;n;bestF.min;bestF.med;bestF.mean;bestF.geom;bestF.max;bestF.sd;lastImprovementFE.min;lastImprovementFE.med;lastImprovementFE.mean;lastImprovementFE.geom;lastImprovementFE.max;lastImprovementFE.sd;lastImprovementTimeMillis.min;lastImprovementTimeMillis.med;lastImprovementTimeMillis.mean;lastImprovementTimeMillis.geom;lastImprovementTimeMillis.max;lastImprovementTimeMillis.sd;totalFEs.min;totalFEs.med;totalFEs.mean;totalFEs.geom;totalFEs.max;totalFEs.sd;totalTimeMillis.min;totalTimeMillis.med;totalTimeMillis.mean;totalTimeMillis.geom;totalTimeMillis.max;totalTimeMillis.sd;goalF;bestFscaled.min;bestFscaled.med;bestFscaled.mean;bestFscaled.geom;bestFscaled.max;bestFscaled.sd;successN;successFEs.min;successFEs.med;successFEs.mean;successFEs.geom;successFEs.max;successFEs.sd;successTimeMillis.min;successTimeMillis.med;successTimeMillis.mean;successTimeMillis.geom;successTimeMillis.max;successTimeMillis.sd;ertFEs;ertTimeMillis;maxFEs;maxTimeMillis
ea1p1_swap2;abz7;4;756;761.5;761.5;761.4899866748019;767;4.509249752822894;8005;9395.5;9182.75;9151.751195919433;9935;853.7393727986702;135;147.5;146.75;146.4445752344153;157;10.90489186863706;10000;10000;10000;10000;10000;0;175;184.5;184.75;184.61439137235973;195;8.180260794538684;656;1.1524390243902438;1.1608231707317074;1.1608231707317074;1.1608079065164663;1.1692073170731707;0.006873856330522731;0;;;;;;;;;;;;;inf;inf;10000;120000
ea1p1_swap2;demo;4;180;180;180;180;180;0;33;73;65.75;61.7025293022418;84;23.879907872519105;1;2;1.75;1.681792830507429;2;0.5;33;73;65.75;61.7025293022418;84;23.879907872519105;2;3;2.75;2.7108060108295344;3;0.5;180;1;1;1;1;1;0;4;33;73;65.75;61.7025293022418;84;23.879907872519105;1;2;1.75;1.681792830507429;2;0.5;65.75;1.75;10000;120000
ea1p1_swap2;la24;4;1015;1028.5;1026.25;1026.2261982741852;1033;8.05708797684788;5218;8308.5;7821.5;7620.464638595248;9451;1932.6562894972642;73;119;117.75;112.23430494843578;160;40.59043401262585;10000;10000;10000;10000;10000;0;134;155.5;157.5;155.97336042658907;185;25.357444666211933;935;1.085561497326203;1.1;1.0975935828877006;1.0975681264964547;1.1048128342245989;0.008617206392350722;0;;;;;;;;;;;;;inf;inf;10000;120000
hc_swap2;abz7;4;804;820;823.5;823.3222584158909;850;19.82422760159901;3798;5612.5;5839.5;5556.776850879124;8335;2102.5303010103485;66;100;104.25;98.56646468667634;151;39.76074278313556;10000;10000;10000;10000;10000;0;175;219;209;207.99300516271921;223;22.80350850198276;656;1.225609756097561;1.25;1.2553353658536586;1.2550644183169068;1.295731707317073;0.030219859148778932;0;;;;;;;;;;;;;inf;inf;10000;120000
hc_swap2;demo;4;180;192.5;192.5;192.22373987227797;205;11.902380714238083;4;33.5;49.75;27.53060177455133;128;53.98996820397903;1;1;1.5;1.3160740129524924;3;1;34;10000;7508.5;2414.736402766418;10000;4983;2;158;120.5;53.49300611903788;164;79.05061669588669;180;1;1.0694444444444444;1.0694444444444444;1.0679096659571;1.1388888888888888;0.0661243373013227;1;34;34;34;34;34;0;1;1;1;1;1;0;30034;481;10000;120000
hc_swap2;la24;4;1065;1114.5;1131.75;1130.1006812239552;1233;71.47668617575012;2130;2471.5;4235.25;3364.07316907124;9868;3759.9463981108383;34;39;60;50.689571138401085;128;45.42392908295509;10000;10000;10000;10000;10000;0;136;139.5;147;146.2564370349321;173;17.644640357154728;935;1.13903743315508;1.1919786096256684;1.210427807486631;1.2086638301860484;1.3187165775401068;0.07644565366390384;0;;;;;;;;;;;;;inf;inf;10000;120000
```


## 5. Evaluating Experiments

The [moptipy](https://thomasweise.github.io/moptipy) system offers a set of tools to evaluate the results collected from experiments.
On one hand, you can [export](#51-exporting-data) the data to formats that can be processed by other tools.
On the other hand, you can plot a variety of different diagrams.
These diagrams can then be [stored](https://thomasweise.github.io/moptipy/moptipy.utils.html#module-moptipy.utils.plot_utils) in different formats, such as `svg` (for the web) or `pdf` (for scientific papers). 

### 5.1. Exporting Data

We already discussed two formats that can be used to export data to Excel or other software tools.

The [End Results CSV format](#42-end-result-csv-files) produces semicolon-separated-values files that include the states of each run.
For every single run, there will be a row with the algorithm name, instance name, and random seed, as well as the best objective value, the last improvement time and FE, and the total time and consumed FEs.

The [End Results Statistics CSV format](#43-end-result-statistics-csv-files) allows you to export statistics aggregated, e.g., over the instance-algorithm combinations, for instance over all algorithms, or for one algorithm over all instances.
The format is otherwise similar to the End Results CSV format. 


### 5.2. Progress Plots

In the file [examples/progress_plot.py](./examples/progress_plot.html), you can find some code running a small experiment and creating "progress plots."
A progress plot is a diagram that shows how an algorithm improves the solution quality over time.
The solution quality can be the raw objective value, the objective value scaled by the goal objective value, or the objective value normalized with the goal objective value.
The time can be measured in objective function evaluations (FEs) or in milliseconds and may be log-scaled or unscaled.
A progress plot can illustrate groups of single runs that were performed in the experiments.
It can also illustrate statistics over the runs, say, the arithmetic mean of the best-so-far objective value at a given point in time.
Both types of data can also be combined in the same diagram.

<a href="https://thomasweise.github.io/moptipy/_static/progress_single_runs_and_mean_f_over_fes.png">
<img alt="Example for a progress plot combining statistics and single runs" src="https://thomasweise.github.io/moptipy/_static/progress_single_runs_and_mean_f_over_fes.png" style="width:70%;max-width:70%;min-width:70%" />
</a>

Progress plots are implemented in the module [moptipy.evaluation.plot_progress_impl](https://thomasweise.github.io/moptipy/moptipy.evaluation.html#module-moptipy.evaluation.plot_progress_impl).


### 5.3. ECDF Plots

In the file [examples/ecdf_plot.py](./examples/ecdf_plot.html), you can find some code running a small experiment and creating "ECDF plots."
The empirical cumulative distribution function (ECDF) is a plot that aggregates data over several runs of an optimization algorithm.
It has the consumed runtime (in FEs or milliseconds) on its horizontal axis and the fraction of runs that succeeded in reaching a specified goal on its vertical axis.
Therefore, an ECDF curve is a monotonously increasing curve:
It remains 0 until the very first (fastest) run of the algorithm reaches the goal, say at time `T1`.
Then, it will increase a bit every single time another run reaches the goal.
At the point in time `T2` when the slowest, last run reaches the goal, it becomes `1`.
Of course, if not all runs reach the goal, it can also remain at a some other level in `[0,1]`.

Let's say we execute 10 runs of our algorithm on a problem instance.
The ECDF remains 0 until the first run reaches the goal.
At this time, it would rise to value `1/10=0.1`.
Once the second run reaches the goal, it will climb to `2/10=0.2`.
If `7` out of our `10` runs can solve the problem and `3` fail to do so, the ECDF would climb to `7/10=0.7` and then remain there.

<a href="https://thomasweise.github.io/moptipy/_static/ecdf_over_log_fes.png">
<img alt="Example for an ECDF plot combining statistics and single runs" src="https://thomasweise.github.io/moptipy/_static/ecdf_over_log_fes.png" style="width:70%;max-width:70%;min-width:70%" />
</a>

ECDF plots are implemented in the module [moptipy.evaluation.plot_ecdf_impl](https://thomasweise.github.io/moptipy/moptipy.evaluation.html#module-moptipy.evaluation.plot_ecdf_impl).


### 5.4. End Results Plot

In the file [examples/end_results_plot.py](./examples/end_results_plot.html), you can find some code running a small experiment and creating "end results plots."
An end results plot is basically a [box plot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.boxplot.html) overlay on top of a [violin plot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.violinplot.html).

Imagine that you conduct multiple runs of one algorithm on one problem instance, let's say 50.
Then you get 50 [log files](#41-log-files) and each of them contains the best solution discovered by the corresponding run.
Now you may want to know how the corresponding 50 objective values are distributed.
You want to get a visual impression about this distribution.
Our end results diagram provide this impression by combining two visualizations:

The [box plot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.boxplot.html) in the foreground shows the
- the median
- the 25% and 75% quantile
- the 95% confidence interval around the median (as notch)
- the arithmetic mean (as a triangle symbol)
- whiskers at the 5% and 95% quantiles, and
- the outliers on both ends of the spectrum.

The [violin plot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.violinplot.html) in the background tries to show the approximate distribution of the values.
A violin plot is something like a smoothed-out, vertical, and mirror-symmetric histogram.
Whereas you can see and compare statistical properties of the end result distribution from the box plots, you cannot really see how they are actually distributed.
For example, it is not clear if the distribution is uni-modal or multi-modal.
You can see this from the violins plotted in the background.

If you compute such plots over multiple algorithm-instance combinations, data will automatically be grouped by problem instance.
This means that the violin-boxes of different algorithms on the same problem will be plotted next to each other.
This, in turn, allows you to easily compare algorithm performance.

In order to make comparing algorithm performance over different instances easier, this plot will use scaled objective values by default.
It will use the goal objective values `g` from the log files to scale all objective values `f` to `f/g`.
Ofcourse you can also use it to plot raw objective values, or even runtimes if you wish.

<a href="https://thomasweise.github.io/moptipy/_static/end_results_scaled.png">
<img alt="Example for an ECDF plot combining statistics and single runs" src="https://thomasweise.github.io/moptipy/_static/end_results_scaled.png" style="width:70%;max-width:70%;min-width:70%" />
</a>

The end result plots are implemented in the module [moptipy.evaluation.plot_end_results_impl](https://thomasweise.github.io/moptipy/moptipy.evaluation.html#module-moptipy.evaluation.plot_end_results_impl).


## 6. License

The copyright holder of this package is Prof. Dr. Thomas Weise (see [Contact](#7-contact)).
The package is licensed under the [GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007](https://github.com/thomasWeise/moptipy/blob/main/LICENSE).


## 7. Contact

If you have any questions or suggestions, please contact
Prof. Dr. [Thomas Weise](http://iao.hfuu.edu.cn/5) (汤卫思教授) of the 
Institute of Applied Optimization (应用优化研究所, [IAO](http://iao.hfuu.edu.cn)) of the
School of Artificial Intelligence and Big Data ([人工智能与大数据学院](http://www.hfuu.edu.cn/aibd/)) at
[Hefei University](http://www.hfuu.edu.cn/english/) ([合肥学院](http://www.hfuu.edu.cn/)) in
Hefei, Anhui, China (中国安徽省合肥市) via
email to [tweise@hfuu.edu.cn](mailto:tweise@hfuu.edu.cn) with CC to [tweise@ustc.edu.cn](mailto:tweise@ustc.edu.cn).
