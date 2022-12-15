"""
Example for bit-string based objective functions.

The problems noted here are mainly well-known benchmark problems from the
discrete optimization community. For these problems, we designed the dedicated
base class :class:`~moptipy.examples.bitstrings.bitstring_problem.\
BitStringProblem`.

The following benchmark problems are provided:

1. :mod:`~moptipy.examples.bitstrings.ising1d`, the one-dimensional
   Ising model, where the goal is that all bits should have the same value as
   their neighbors.
2. The :mod:`~moptipy.examples.bitstrings.jump` problem is equivalent
   to :mod:`~moptipy.examples.bitstrings.onemax`, but has a deceptive
   region right before the optimum.
3. The :mod:`~moptipy.examples.bitstrings.leadingones` problem,
   where the goal is to find a bit string with the maximum number of leading
   ones.
4. The :mod:`~moptipy.examples.bitstrings.onemax` problem, where the
   goal is to find a bit string with the maximum number of ones.
5. The :mod:`~moptipy.examples.bitstrings.trap` problem, which is like
   OneMax, but with the optimum and worst-possible solution swapped. This
   problem is therefore highly deceptive.
6. The :mod:`~moptipy.examples.bitstrings.w_model`, a benchmark
   problem with tunable epistasis, uniform neutrality, and
   ruggedness/deceptiveness.
7. The :mod:`~moptipy.examples.bitstrings.zeromax` problem, where the
   goal is to find a bit string with the maximum number of zeros. This is the
   opposite of the OneMax problem.
"""
