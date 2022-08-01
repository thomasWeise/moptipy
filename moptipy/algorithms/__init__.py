"""
In this package, we provide implementations of metaheuristic algorithms.

We divide them into single-objective algorithms, which are provided in the
package :mod:`moptipy.algorithms.so`, and multi-objective algorithms,
which can be found in package :mod:`moptipy.algorithms.mo`.

Methods which are based on random sampling and random walks are unaffected by
the number of objective functions and therefore can be found in this package
directly.

The black-box methods are given directly in these package, more specialized
algorithms will be placed in sub-packages corresponding to their requirements
for the search- and solution space.
"""
