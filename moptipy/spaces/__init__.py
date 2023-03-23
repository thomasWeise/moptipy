"""
In this package, some pre-defined search and solution spaces are provided.

These spaces implement the base class :class:`~moptipy.api.space.Space`, which
defines the API for creating, copying, storing, and comparing (for equality)
of data structures that can be used to represent either points in the search
space or candidate solutions in the solution space.

The following pre-defined spaces are currently available:

- :class:`~moptipy.spaces.bitstrings.BitStrings`, the space of `n`-dimensional
  bit strings
- :class:`~moptipy.spaces.intspace.IntSpace`, a space of `n`-dimensional
  integer strings, where each element is between (and including) a minimum
  and a maximum value (inclusive)
- :class:`~moptipy.spaces.permutations.Permutations` is a special version of
  the :class:`~moptipy.spaces.intspace.IntSpace` where all elements are
  permutations of a base string
  :attr:`~moptipy.spaces.permutations.Permutations.blueprint`. This means that
  it can represent permutations both with and without repetitions. Depending
  on the base string, each element may occur an element-specific number of
  times. For the base string `(-1, -1, 2, 7, 7, 7)`, for example, `-1` may
  occur twice, `2` can occur once, and `7` three times.
- :class:`~moptipy.spaces.ordered_choices.OrderedChoices` is a combination of
  permutations and combinations. There are `n` choices of one or multiple
  different values each. The choices are either disjoint or identical. An
  element from the space picks one value per choice. The order of the elements
  matters.
- :class:`~moptipy.spaces.vectorspace.VectorSpace` is the space of
  `n`-dimensional floating point number vectors
"""
