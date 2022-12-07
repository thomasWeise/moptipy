"""
Example for bit-string based objective functions.

The problems noted here are mainly well-known benchmark problems from the
discrete optimization community. For these problems, we designed the dedicated
base class :class:`~moptipy.examples.bitstrings.bitstring_problem.\
BitStringProblem`.

The following benchmark problems are provided:

1. :class:`~moptipy.examples.bitstrings.ising1d.Ising1d`, the one-dimensional
   Ising model, where the goal is that all bits should have the same value as
   their neighbors.
2. The :class:`~moptipy.examples.bitstrings.leadingones.LeadingOnes` problem,
   where the goal is to find a bit string with the maximum number of leading
   ones.
3. The :class:`~moptipy.examples.bitstrings.onemax.OneMax` problem, where the
   goal is to find a bit string with the maximum number of ones.
4. The :class:`~moptipy.examples.bitstrings.trap.Trap` problem, which is like
   OneMax, but with the optimum and worst-possible solution swapped. This
   problem is therefore highly deceptive.
5. The :class:`~moptipy.examples.bitstrings.w_model.WModel`, a benchmark
   problem with tunable epistasis, uniform neutrality, and ruggedness/
   deceptiveness.
6. The :class:`~moptipy.examples.bitstrings.zeromax.ZeroMax` problem, where the
   goal is to find a bit string with the maximum number of zeros. This is the
   opposite of the OneMax problem.
"""
