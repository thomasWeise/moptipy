"""
In this package we provide operators for bit strings.

- Module :mod:`~moptipy.operators.bitstrings.op0_random` offers a nullary
  search operator that samples bit strings of a fixed length in a uniform
  random fashion.
- Module :mod:`~moptipy.operators.bitstrings.op1_flip1` offers a unary
  operator that flips exactly one bit.
- Module :mod:`~moptipy.operators.bitstrings.op1_flip_m` offers a unary
  operator with step size (see
  :class:`~moptipy.api.operators.Op1WithStepSize`). A step size of `0.0`
  means that it will flip one bit, a step size of `1.0` means that it flips
  all bits.
- Module :mod:`~moptipy.operators.bitstrings.op1_m_over_n_flip` flips each
  bit with probability `m/n` but ensures that at least one bit is flipped.
  This results in a number of bits being flipped being drawn from a Binomial
  distribution (and re-drawn if `0` was drawn).
- Module :mod:`~moptipy.operators.bitstrings.op2_uniform` offers the binary
  uniform crossover operation which copies each bit with probability `0.5`
  from the first and with the same probability from the second parent.
"""
