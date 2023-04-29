"""
Search operators for permutations with repetitions.

These operators can be used in conjunction with the space
:class:`~moptipy.spaces.permutations.Permutations`.

- Module :mod:`~moptipy.operators.permutations.op0_shuffle` provides a nullary
  operator sampling an entirely random permutation.
- Module :mod:`~moptipy.operators.permutations.op1_insert1` provides a unary
  operator that extracts one element and inserts it elsewhere.
- Module :mod:`~moptipy.operators.permutations.op1_swap2` provides a unary
  operator which swaps exactly two different elements of a permutation.
- Module :mod:`~moptipy.operators.permutations.op1_swapn` provides a unary
  operator which tries to find a cyclic move whose length would be
  approximately binomially distributed, i.e., which may swap two elements with
  probability 0.5, three elements with probability 0.25, 4 elements with
  probability 0.125, and so on. It does so by performing a sequence of swaps
  but does not check whether one swap undoes a previous one.
- Module :mod:`~moptipy.operators.permutations.op1_swap_exactly_n` offers a
  unary operator with a step size (see
  :class:`~moptipy.api.operators.Op1WithStepSize`). A step size of `0.0` means
  that it will swap two elements, a step size of `1.0` means that it tries to
  perform the largest-possible modification. If it is applied to permutations
  where each element occurs once, this means that all elements change their
  position. If applied to permutations with repetitions, things are more
  complex and some moves may be impossible, in which case a best effort is
  attempted.
- Module :mod:`~moptipy.operators.permutations.op1_swap_try_n` is similar to
  the operator from :mod:`~moptipy.operators.permutations.op1_swap_exactly_n`,
  but invests less effort into reaching the prescribed number of modifications
  exactly. Instead, it will accept making only fewer swaps more easily.
- Module :mod:`~moptipy.operators.permutations.op2_gap` offers an operator
  that tries to build a new permutation by appending not-yet-appended elements
  from both input permutations, alternating between them randomly.
- Module :mod:`~moptipy.operators.permutations.op2_ox2` implements a slightly
  modified version of the order-based crossover operator for permutations.
"""
