[![make build](https://github.com/thomasWeise/moptipy/actions/workflows/build.yaml/badge.svg)](https://github.com/thomasWeise/moptipy/actions/workflows/build.yaml)

# moptipy: Metaheuristic Optimization in Python

This will be a library with implementations of metaheuristic optimization methods in Python.
Well, it will eventually be, because I first need to learn Python.

- [Introduction](#1-introduction)
- [Installation](#2-installation)
- [How-Tos](#3-how-to)
  - [Solving Optimization Problems](#31-how-to-solve-an-optimization-problem)
- [Data Formats](#4-data-formats)
  - [Log Files](#41-log-files)
  - [End Results CSV Files](#42-end-result-csv-files)
  - [End Result Statistics CSV Files](#43-end-result-statistics-csv-files)
- [License](#5-license)
- [Contact](#6-contact)

## 1. Introduction

Metaheuristic optimization algorithms are methods for solving hard problems.
Here we provide an API that can be used to implement them and to experiment with them.
The codes in this repository are ued as examples in the book [Optimization Algorithms](https://thomasweise.github.io/oa/) which I am currently writing.
Its full sources are available on GitHub at <https://github.com/thomasWeise/oa>.

## 2. Installation

You can easily install this library using `pip` by doing

```
pip install git+https://github.com/thomasWeise/moptipy.git
```

## 3. How-To

You can find many examples of how to use the [moptipy](https://thomasweise.github.io/moptipy) library in the folder `examples`.

### 3.1. How to Solve an Optimization Problem

If you want to solve an optimization problem with [moptipy](https://thomasweise.github.io/moptipy), then you need at least the following three things:

1. a space `Y` of possible solutions,
2. an objective function `f`  rating the solutions, i.e., which maps elements `y` of `Y` to either integer or float numbers, where *smaller* values are better, and
3. an optimization algorithm that navigates through `Y` and tries to find solutions `y` in `Y` with low corresponding values `f(y)`.

You may need more components, but if you have these three, then you can run an experiment.

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

- *Semicolon-separated values* can hold a series of data values, where each row is divided into multiple values and the values are separated by `;`
- *Key-values* sections represent, well, values for keys in form of a mapping compatible with [YAML](https://yaml.org/spec/1.2/spec.html#mapping).
  In other words, each line contains a key, followed by `: `, followed by the value.
  The keys can be hierarchically structured in scopes, for example `a.b` and `a.c` indicate two keys `b` and `c` that belong to scope `a`.
- *Raw text* sections contain text without a general or a priori structure, e.g., the string representation of the best solutions found.


##### The Section `PROGRESS`

When setting up an algorithm execution, you can specify whether you want to log the progress of the algorithm.
If and only if you choose to log the progress, the `PROGRESS` section will be contained in the log file.
You can also choose if you want to log all algorithm steps or only the improving moves, the latter being the default behavior.
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
Usually, we could log one data point for every improvement of the objective value.


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

- process wrapper parameters (scope `p`):
  - `p.name`: the name of the process wrapper, i.e., a short mnemonic describing its purpose
  - `p.class`: the python class of the process wrapper
  - `p.maxTimeMillis`: the maximum clock time in milliseconds, if specified
  - `p.maxFEs`: the maximum number of objective function evaluations (FEs), if specified
  - `p.goalF`: the goal objective value, if specified
  - `p.randSeed`: the random seed in decimal notation
  - `p.randSeed(hex)`: the random seed in hexadecimal notation
  - `p.randGenType`: the class of the random number generator
  - `p.randBitGenType`: the class of the bit generator used by the random number generator
- algorithm parameters: scope `a`, includes algorithm `name`, `class`, etc.
- solution space scope `y`, includes `name` and `class` of solution space
- objective function information: scope `f`
- search space information (if search space is different from solution space): scope `x`
- encoding information (if encoding is defined): scope `g` 


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

You can execute the following Python code to obtain an example log file:

```python
from moptipy.algorithms.ea1plus1 import EA1plus1  # the algorithm we use
from moptipy.examples.jssp.experiment import run_experiment  # the JSSP runner
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle  # 0-ary op
from moptipy.operators.permutations.op1_swap2 import Op1Swap2  # 1-ary op
from moptipy.utils.temp import TempDir  # temp directory tool

# We work in a temporary directory, i.e., delete all generated files on exit.
# For a real experiment, you would put an existing directory path in `td`
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
os.version: #40-Ubuntu SMP Mon Mar 7 08:03:10 UTC 2022
END_SYS_INFO
BEGIN_RESULT_Y
1,20,30,0,30,40,3,145,165,2,170,180,1,0,20,0,40,60,2,60,80,3,165,180,2,0,30,0,60,80,1,80,130,3,130,145,1,30,60,3,60,90,0,90,130,2,130,170,3,0,50,2,80,92,1,130,160,0,160,170
END_RESULT_Y
BEGIN_RESULT_X
2,1,3,1,0,0,2,0,1,2,3,1,0,2,1,3,0,3,2,3
END_RESULT_X
```


### 4.2. End Result CSV Files

While a log file contains all the data of a single run, you often want to get just the basic measurements, such as the result objective values, from all runs of one experiment in a single file.
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

### 4.2.2. An Example for End Results Files

Let us execute an abridged example experiment, parse all log files, condense their information into an end results statistics file, and then print that file's contents.
We can do that as follows

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
# and not use the `with` block.
with TempDir.create() as td:
    run_experiment(  # run the JSSP experiment with the following parameters:
        base_dir=td,  # base directory to write all log files to
        algorithms=[  # the set of algorithm generators
            lambda inst, pwr: EA1plus1(Op0Shuffle(pwr), Op1Swap2()),  # algo 1
            lambda inst, pwr: HillClimber(Op0Shuffle(pwr), Op1Swap2())],  # 2
        instances=("demo", "abz7", "la24"),  # we use 3 JSSP instances
        max_fes=10000,  # we grant 10000 FEs per run
        n_runs=4)  # perform 4 runs per algorithm * instance combination

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

We can basically execute the same abridged experiment as in the previous section, but now take the aggregation of information one step further:

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
# and not use the `with` block.
with TempDir.create() as td:
    run_experiment(  # run the JSSP experiment with the following parameters:
        base_dir=td,  # base directory to write all log files to
        algorithms=[  # the set of algorithm generators
            lambda inst, pwr: EA1plus1(Op0Shuffle(pwr), Op1Swap2()),  # algo 1
            lambda inst, pwr: HillClimber(Op0Shuffle(pwr), Op1Swap2())],  # 2
        instances=("demo", "abz7", "la24"),  # we use 3 JSSP instances
        max_fes=10000,  # we grant 10000 FEs per run
        n_runs=4)  # perform 4 runs per algorithm * instance combination

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


## 5. License

The copyright holder of this package is Prof. Dr. Thomas Weise (see Contact).
The package is licensed under the GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007.


## 6. Contact

If you have any questions or suggestions, please contact
[Prof. Dr. Thomas Weise](http://iao.hfuu.edu.cn/5) of the
[Institute of Applied Optimization](http://iao.hfuu.edu.cn/) at
[Hefei University](http://www.hfuu.edu.cn) in
Hefei, Anhui, China via
email to [tweise@hfuu.edu.cn](mailto:tweise@hfuu.edu.cn) with CC to [tweise@ustc.edu.cn](mailto:tweise@ustc.edu.cn).
