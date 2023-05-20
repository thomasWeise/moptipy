"""
Search operators for signed permutations with repetitions.

These operators can be used in conjunction with the space
:mod:`~moptipy.spaces.signed_permutations`.

- Module :mod:`~moptipy.operators.signed_permutations.op0_shuffle_and_flip`
  provides a nullary operator sampling an entirely random signed permutation.
- Module :mod:`~moptipy.operators.signed_permutations.op1_swap_2_or_flip`
  provides a unary operator which either swaps exactly two different elements
  of a permutation *or* flips the sign of one element.
"""
