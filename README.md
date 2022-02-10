[![make build](https://github.com/thomasWeise/moptipy/actions/workflows/build.yaml/badge.svg)](https://github.com/thomasWeise/moptipy/actions/workflows/build.yaml)

# moptipy: Metaheuristic Optimization in Python

This will be a library with implementations of metaheuristic optimization methods in Python.
Well, it will eventually be, because I first need to learn Python.


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


## 3. Data Formats

We develop several data formats to store and evaluate the results of computational experiments with our `moptipy` software.
Here you can find their basic definitions.


### 3.1. Log Files

#### 3.1.1. File Names and Folder Structure

One independent run of an algorithm on one problem instance produces one log file.
Each run is identified by the algorithm that is applied, the problem instance to which it is applied, and the random seed.
This tuple is reflected in the file name.
`ea1p1_swap2_demo_0x5a9363100a272f12.txt`, for example, represents the algorithm `ea1p1_swap2` applied to the problem instance `demo` and started with random seed `0x5a9363100a272f12` (where `0x` stands for hexademical notation).
The log files are grouped in a `algorithm`-`instance` folder structure.
In the above example, there would be a folder `ea1p1_swap2` containing a folder `demo`, which, in turn, contains all the log files from all runs of that algorithm on this instance.


#### 3.1.2. Log File Sections

A log file is a simple text file divided into several sections.
Each section `X` begins with the line `BEGIN_X` and ends with the line `END_X`.
There are three types of sections:

- *Semicolon-separated values* can hold a series of data values, where each row is divided into multiple values and the values are separated by `;`
- *Key-values* sections represent, well, values for keys in form of a mapping compatible with [YAML](https://yaml.org/spec/1.2/spec.html#mapping).
  In other words, each line contains a key, followed by `: `, followed by the value.
  The keys can be hierarchically structured in scopes, for example `a.b` and `a.c` indicate two keys `b` and `c` that belong to scope `a`.
- *Raw text* sections contain text without a general or a priori structure, e.g., the string representation of the best solutions found.


##### The Section `PROGRESS`

The `Progress` section contains log points describing the algorithm progress over time in a semicolon-separated values format with one data point per line.
It has an internal header describing the data columns.
There will at least be the following columns:

1. `fes` denoting the integer number of performed objective value evaluations
2. `timeMS` the clock time that has passed since the start of the run, measured in milliseconds and stored as integer value
3. `f` the best-so-far objective value

This configuration is denoted by the header `fes;timeMS;f`.
After this header and until `END_PROGRESS`, each line will contain one data point with values for the specified columns.
Notice that for each FE, there will be at most one data point but there might be multiple data points per millisecond.
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
  - `p.name`: the name of the process wrapper
  - `p.type`: the python class of the process wrapper
  - `p.maxTimeMillis`: the maximum clock time in milliseconds, if specified
  - `p.maxFEs`: the maximum number of objective function evaluations (FEs), if specified
  - `p.goalF`: the goal objective value, if specified
  - `p.randSeed`: the random seed in decimal notation
  - `p.randSeed(hex)`: the random seed in hexadecimal notation
  - `p.randGenType`: the class of the random number generator
  - `p.randBitGenType`: the class of the bit generator used by the random number generator
- algorithm parameters: scope `a`, includes algorithm name, class, etc.
- solution space scope `y`, includes name and class of solution space
- objective function information: scope `f`
- search space information (if search space is different from solution space): scope `x`
- encoding information (if encoding is defined): scope `g` 


#### 3.1.3. Example

You can execute the following Python code to obtain an example log file:

```python
from moptipy.algorithms.ea1p1 import EA1p1
from moptipy.examples.jssp.experiment import run_experiment
from moptipy.operators.pwr.op0_shuffle import Op0Shuffle
from moptipy.operators.pwr.op1_swap2 import Op1Swap2
from moptipy.utils.temp import TempDir

with TempDir.create() as td:
    run_experiment(base_dir=td, algorithms=[lambda inst, pwr: EA1p1(Op0Shuffle(pwr), Op1Swap2())], instances=("demo", ), n_runs=1, n_threads=1)
    print(td.resolve_inside("ea1p1_swap2/demo/ea1p1_swap2_demo_0x5a9363100a272f12.txt").read_all_str())
```

The example log file printed by the above code will then look as follows:

```
BEGIN_PROGRESS
fes;timeMS;f
1;1;270
8;1;235
12;1;230
17;2;210
32;2;185
66;2;180
END_PROGRESS
BEGIN_STATE
totalFEs: 66
totalTimeMillis: 2
bestF: 180
lastImprovementFE: 66
lastImprovementTimeMillis: 2
END_STATE
BEGIN_SETUP
p.name: LoggingProcessWithSearchSpace
p.type: <class 'moptipy.api._process_ss_log._ProcessSSLog'>
p.maxTimeMillis: 120000
p.goalF: 180
p.randSeed: 6526669205530947346
p.randSeed(hex): 0x5a9363100a272f12
p.randGenType: <class 'numpy.random._generator.Generator'>
p.randBitGenType: <class 'numpy.random._pcg64.PCG64'>
a.name: ea1p1_swap2
a.type: <class 'moptipy.algorithms.ea1p1.EA1p1'>
a.op0.name: shuffle
a.op0.type: <class 'moptipy.operators.pwr.op0_shuffle.Op0Shuffle'>
a.op1.name: swap2
a.op1.type: <class 'moptipy.operators.pwr.op1_swap2.Op1Swap2'>
y.name: gantt_demo
y.type: <class 'moptipy.examples.jssp.gantt_space.GanttSpace'>
y.inst.name: demo
y.inst.type: <class 'moptipy.examples.jssp.instance.Instance'>
y.inst.machines: 5
y.inst.jobs: 4
y.inst.makespanLowerBound: 180
y.inst.makespanUpperBound: 482
f.name: makespan
f.type: <class 'moptipy.examples.jssp.makespan.Makespan'>
x.name: perm4w5r
x.type: <class 'moptipy.spaces.permutationswr.PermutationsWithRepetitions'>
x.nvars: 20
x.dtype: b
x.min: 0
x.max: 3
x.repetitions: 5
g.name: operation_based_encoding
g.type: <class 'moptipy.examples.jssp.ob_encoding.OperationBasedEncoding'>
END_SETUP
BEGIN_SYS_INFO
session.start: 2022-02-10 16:35:37.630541
session.node: home
session.ip_address: 10.8.0.6
version.moptipy: 0.8.1
version.numpy: 1.21.5
version.numba: 0.55.1
version.matplotlib: 3.5.1
version.psutil: 5.9.0
version.scikitlearn: 1.0.2
hardware.machine: x86_64
hardware.n_physical_cpus: 8
hardware.n_logical_cpus: 16
hardware.cpu_mhz: (2200MHz..3700MHz)*16
hardware.byteorder: little
hardware.cpu: AMD Ryzen 7 2700X Eight-Core Processor
hardware.mem_size: 16719241216
python.version: 3.9.7 (default, Sep 10 2021, 14:59:43) [GCC 11.2.0]
python.implementation: CPython
os.name: Linux
os.release: 5.13.0-28-generic
os.version: #31-Ubuntu SMP Thu Jan 13 17:41:06 UTC 2022
END_SYS_INFO
BEGIN_RESULT_Y
0,10,20,30,170,180,125,145,20,40,0,20,40,60,145,160,40,60,60,110,0,30,110,125,90,130,30,60,130,170,60,90,130,140,140,170,60,72,0,50
END_RESULT_Y
BEGIN_RESULT_X
2,0,1,1,0,1,3,2,3,0,1,3,3,2,0,3,2,2,0,1
END_RESULT_X
```


## 4. License

The copyright holder of this package is Prof. Dr. Thomas Weise (see Contact).
The package is licensed under the GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007.


## 5. Contact

If you have any questions or suggestions, please contact
[Prof. Dr. Thomas Weise](http://iao.hfuu.edu.cn/5) of the
[Institute of Applied Optimization](http://iao.hfuu.edu.cn/) at
[Hefei University](http://www.hfuu.edu.cn) in
Hefei, Anhui, China via
email to [tweise@hfuu.edu.cn](mailto:tweise@hfuu.edu.cn) with CC to [tweise@ustc.edu.cn](mailto:tweise@ustc.edu.cn).
