"""
The API for implementing optimization algorithms and -problems.

This package provides two things:

* the basic, abstract API for implementing optimization algorithms and
  problems
* the abstraction and implementation of black-box processes in form of
  the :mod:`~moptipy.api.process` API and its implementations

The former helps us to implement and plug together different components of
optimization problems and optimization algorithms. The latter allows us to
apply algorithms to problems and to collect the results in a transparent way.
It also permits logging of the algorithm progress or even the change of
dynamic parameters.

The `moptipy` API has the following major elements for implementing
optimization algorithms and their components. You are encouraged to read more
about the logic behind this structure in our free online book at
https://thomasweise.github.io/oa.

- :mod:`~moptipy.api.component` provides the base class
  :class:`~moptipy.api.component.Component` from which all components of
  algorithm and optimization problems inherit. It offers the basic methods
  :meth:`~moptipy.api.component.Component.log_parameters_to` for writing all
  parameters and configuration elements of a component to a
  :class:`~moptipy.utils.logger.KeyValueLogSection` of a log file (embodied by
  an instance of :class:`~moptipy.utils.logger.Logger`) and the method
  :meth:`~moptipy.api.component.Component.initialize`, which must be
  overwritten to do any initialization necessary before a run of the
  optimization process.
- :mod:`~moptipy.api.algorithm` provides the base class
  :class:`~moptipy.api.algorithm.Algorithm` from which any optimization
  algorithm will inherit and which defines the optimization algorithm
  API, most importantly the method
  :meth:`~moptipy.api.algorithm.Algorithm.solve` which must be overwritten to
  implement the algorithm behavior. Here you can also find subclasses such as
  :class:`~moptipy.api.algorithm.Algorithm0`,
  :class:`~moptipy.api.algorithm.Algorithm1`, and
  :class:`~moptipy.api.algorithm.Algorithm2` for algorithms that use nullary,
  nullary and unary, as well as nully, unary, and binary search operators.
- :mod:`~moptipy.api.encoding` provides the base class
  :class:`~moptipy.api.encoding.Encoding` which can be used to implement the
  decoding procedure :meth:`~moptipy.api.encoding.Encoding.decode` which
  translates elements of a search space to elements of the solution space.
  Such an encoding may be needed if the two spaces differ, as they do, for
  instance, in our Job Shop Scheduling Problem example (see
  :mod:`~moptipy.examples.jssp.ob_encoding`).
- :mod:`~moptipy.api.mo_algorithm` provides the base class
  :class:`~moptipy.api.mo_algorithm.MOAlgorithm` for multi-objective
  optimization algorithms, which is a subclass of
  :class:`~moptipy.api.algorithm.Algorithm`.
- :mod:`~moptipy.api.mo_archive` provides the base class
  :class:`~moptipy.api.mo_archive.MOArchivePruner` for pruning archives of
  non-dominated solutions during a multi-objective optimization process. Since
  multi-objective optimization problems may have many possible solutions, but
  we can only return/retain a finite number of them, it may be necessary to
  decide which solution to keep and which to dispose of. This is the task
  of the archive pruner.
- :mod:`~moptipy.api.objective` provides the base class
  :class:`~moptipy.api.objective.Objective` for objective functions, i.e., for
  implementing the criteria rating how good a solution is. Objective functions
  are subject to minimization and provide a method
  :meth:`~moptipy.api.objective.Objective.evaluate` which computes an integer
  or floating point value rating one solution. They may additionally have
  a :meth:`~moptipy.api.objective.Objective.lower_bound` and/or an
  :meth:`~moptipy.api.objective.Objective.upper_bound`.
- :mod:`~moptipy.api.mo_problem` provides the base class
  :class:`~moptipy.api.mo_problem.MOProblem`, which represents a
  multi-objective optimization problem. This is a subclass of
  :class:`~moptipy.api.objective.Objective` *and* at the same time, a
  collection of multiple instances of
  :class:`~moptipy.api.objective.Objective`. Each multi-objective problem can
  compute a vector of objective values via
  :meth:`~moptipy.api.mo_problem.MOProblem.f_evaluate`, but also implements
  the single-objective :meth:`~moptipy.api.mo_problem.MOProblem.evaluate`
  method returning a default scalarization of the objective vector. This makes
  multi-objective optimization compatible with single-objective optimization.
  While actual multi-objective algorithms can work in a truly multi-objective
  fashion, logging can be unified and even single-objective methods can be
  applied. To allow for the representation of preferences, the default
  Pareto domination relationship can be overwritten by providing a custom
  implementation of :meth:`~moptipy.api.mo_problem.MOProblem.f_dominates`.
- :mod:`~moptipy.api.operators` provides the base classes
  :class:`~moptipy.api.operators.Op0`, :class:`~moptipy.api.operators.Op1`, and
  :class:`~moptipy.api.operators.Op2` fur nullary, unary, and binary search
  operators, respectively. These can be used to implement the methods
  :meth:`~moptipy.api.operators.Op0.op0`,
  :meth:`~moptipy.api.operators.Op1.op1`, and
  :meth:`~moptipy.api.operators.Op2.op2` that are used by metaheuristic
  optimization algorithms to sample solutions and to derive solutions from
  existing ones.
- :mod:`~moptipy.api.space` provides the base class
  :class:`~moptipy.api.space.Space` for implementing the functionality of
  search and solution spaces. An instance of :class:`~moptipy.api.space.Space`
  offers methods such as :meth:`~moptipy.api.space.Space.create` for creating
  a data structure for holding point in the search space (with undefined
  contents), :meth:`~moptipy.api.space.Space.copy` for copying one data
  structure to another one, :meth:`~moptipy.api.space.Space.to_str` and
  :meth:`~moptipy.api.space.Space.from_str` to convert a data structure to
  and from a string, :meth:`~moptipy.api.space.Space.is_equal` to check
  whether one data structure equals another one, and
  :meth:`~moptipy.api.space.Space.validate` to verify whether the contents
  of a data structure are valid. With these methods, optimization algorithms
  can create and copy the data structure containers to hold solutions as they
  need without requiring any information about the actual contents and
  layout of these structures. The search operators (see above) then can handle
  the actual processing of the data structures. At the same time, the string
  conversion routines allow for storing the results of algorithms in log
  files.

The algorithm and experiment execution API has the following major elements:

- :mod:`~moptipy.api.execution` provides the builder class
  :class:`~moptipy.api.execution.Execution` that is used to construct one
  application of an optimization algorithm to an optimization problem. It
  configures the termination criteria, what information should be logged, and
  which algorithm to apply to which problem. Its method
  :meth:`~moptipy.api.execution.Execution.execute` returns an instance of
  :class:`~moptipy.api.process.Process` that contains the state of the
  optimization *after* it is completed, i.e., after the algorithm was
  executed. Log files follow the specification at
  https://thomasweise.github.io/moptipy/#log-file-sections.
- The module :mod:`~moptipy.api.logging` holds mainly string constants that
  identify sections and keys for the data to be written to log files.
- :mod:`~moptipy.api.mo_execution` provides the builder class
  :class:`~moptipy.api.mo_execution.MOExecution`, which is the multi-objective
  equivalent of :class:`~moptipy.api.execution.Execution`.
- :mod:`~moptipy.api.experiment` offers the function
  :func:`~moptipy.api.experiment.run_experiment` which allows you to execute
  a structured experiment applying a set of optimization algorithms to a set
  of optimization problems in a reproducible fashion, in parallel or in a
  distributed way. It will create a folder structure as prescribed in
  https://thomasweise.github.io/moptipy/#file-names-and-folder-structure.
  The data from such a folder structure can then be read in by the experiment
  evaluation tools in package :mod:`~moptipy.evaluation` and discussed in
  https://thomasweise.github.io/moptipy/#evaluating-experiments.
- :mod:`~moptipy.api.process` provides the class
  :class:`~moptipy.api.process.Process`, which is the basis for all
  optimization processes. It offers the functionality of an objective
  function, the search space, the termination criterion, and the random
  number generator to the optimization algorithm. At the same time, it
  collects the best-so-far solution and writes log files (if needed) for the
  user.
- :mod:`~moptipy.api.mo_process` provides the class
  :class:`~moptipy.api.mo_process.MOProcess`, which is the multi-objective
  equivalent of :class:`~moptipy.api.process.Process`. It also collects an
  archive of solutions.
- :mod:`~moptipy.api.subprocesses` offers several functions to slice off
  computational budget of a process to allow an algorithm to use a
  sub-algorithm, to wrap processes for running external algorithms that do
  not respect the `moptipy` API, or to run sub-algorithms starting with a
  specified initial solution.
"""
