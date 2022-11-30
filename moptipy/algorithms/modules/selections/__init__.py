"""
The set of selection algorithms.

:class:`~moptipy.algorithms.modules.selection.Selection` algorithms are
algorithms that choose elements from a pool of records based on their
:attr:`~moptipy.algorithms.modules.selection.FitnessRecord.fitness` and
random numbers. They are commonly used in Evolutionary Algorithms
(:class:`~moptipy.algorithms.so.general_ea.GeneralEA`).

Currently, the following selection algorithms have been implemented:

- :class:`~moptipy.algorithms.modules.selections.best.Best` selection picks
  the `n` best solutions from an array. It is the survival selection scheme
  used in a (mu+lambda) :class:`~moptipy.algorithms.so.ea.EA`.
- :class:`~moptipy.algorithms.modules.selections.random_without_repl\
.RandomWithoutReplacement` randomly chooses `n` solutions without replacement.
  This is the mating selection scheme used in a (mu+lambda)
  :class:`~moptipy.algorithms.so.ea.EA`.
- :class:`~moptipy.algorithms.modules.selections.fitness_proportionate_sus\
.FitnessProportionateSUS` performs fitness proportionate selection for
  minimization using Stochastic Uniform Sampling. It also allows you to
  specify the selection probability for the worst element.
- :class:`~moptipy.algorithms.modules.selections.tournament_with_repl.\
TournamentWithReplacement`
  performs tournament selection with a specified tournament size with
  replacement.
- :class:`~moptipy.algorithms.modules.selections.tournament_without_repl.\
TournamentWithoutReplacement`
  performs tournament selection with a specified tournament size without
  replacement.
"""
