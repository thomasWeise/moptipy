"""A nullary operator filling a :mod:`~moptipy.spaces.ordered_choices`."""
from typing import Callable, Final

import numpy as np
from numpy.random import Generator

from moptipy.api.operators import Op0
from moptipy.spaces.ordered_choices import OrderedChoices
from moptipy.utils.types import type_error


class Op0ChooseAndShuffle(Op0):
    """Randomly initialize :mod:`~moptipy.spaces.ordered_choices` elements."""

    def __init__(self, space: OrderedChoices):
        """
        Initialize this shuffle operation: use blueprint from space.

        :param space: the search space
        """
        super().__init__()
        if not isinstance(space, OrderedChoices):
            raise type_error(space, "space", OrderedChoices)
        #: the internal blueprint for filling permutations
        self.__blueprint: Final[np.ndarray] = space.blueprint
        #: the internal choices
        self.__choices: Final[Callable[[int], tuple[int, ...]]] \
            = space.choices.__getitem__

    def op0(self, random: Generator, dest: np.ndarray) -> None:
        """
        Fill the destination with random choices and shuffle it.

        :param random: the random number generator
        :param dest: the destination array
        """
        np.copyto(dest, self.__blueprint)  # Copy blueprint to dest.
        random.shuffle(dest)  # Shuffle destination array randomly.

        # Now we replace the elements in the shuffled destination array
        # with random selections.
        ri: Final[Callable[[int], int]] = random.integers  # fast call
        choices: Final[Callable[[int], tuple[int, ...]]] = self.__choices
        for i, e in enumerate(dest):
            source = choices(e)
            dest[i] = source[ri(len(source))]

    def __str__(self) -> str:
        """
        Get the name of this operator.

        :return: "chooseAndShuffle"
        """
        return "chooseAndShuffle"
