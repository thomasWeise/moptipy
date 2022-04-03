"""
The API for implementing optimization algorithms and -problems.

This package provides two things:

* the basic, abstract API for implementing optimization algorithms and
  problems
* the abstraction and implementation of black-box processes in form of
  :py:class:`~moptipy.api.process.Process` and its implementations

The former helps us to implement and plug together different components
of optimization problems and optimization algorithms.
The latter allows us to apply algorithms to problems and to collect the
results in a transparent way.
It also permits logging of the algorithm progress or even the change of
dynamic parameters.
"""
