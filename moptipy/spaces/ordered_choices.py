"""
Permutations of a selection (or ordered choices) as strings.

Imagine the following situation: We have a rectangle (two-dimensional) and
want to place `n` other, smaller rectangles into it. Let each rectangle have
a unique ID from `1` to `n`. We can use a permutation of `1..n` to represent
the order in which we insert the small rectangles. However, maybe we want to
also be able to rotate a rectangle by 90° and then place it. So additionally
to the insertion order, we may want to remember whether the rectangle is
placed as-is or rotated. We could do this with a "permutation of choices",
where we can place either `i` or `-i` for `i` from `1..n`. A negative value
could mean "insert rotated by 90°" whereas a positive value means "insert as
is". So we have `n` (disjoint) choices, each of which with two options. From
each choice, we can pick one value. Then the order in which the values appear
marks the insertion order. So this is basically a "super space" of
permutations, as it deals both with the order of the elements and their values
(resulting from the selected choice).

A string consists of `n` elements. There also are `n` so-called "selections,"
each of which being a set offering a choice from  different values. Any two
selections must either be entirely disjoint or equal. The final string must
contain one value from each selection.

Let's say that we have three selections, e.g., `[1, 2, 3]`, `[4, 5]`, and
`[6]`. Then the "selection permutations" space contains, e.g., the string
`[4, 3, 6]` or `[2, 5, 6]` -- one value from each selection. It does not
contain `[1, 3, 5]`, though, because that string has two values from the first
selection.

This space is a super space of the :mod:`~moptipy.spaces.permutations`, i.e.,
the space of permutations with repetitions. Sometimes (check
:meth:`OrderedChoices.is_compatible_with_permutations`), the search
operators defined in package :mod:`~moptipy.operators.permutations` can also
be applied to the elements of our space here, although they may not be able
to explore the space in-depth (as they will not alter the choices and only
permute the chosen elements).
"""
from collections import Counter
from math import factorial
from typing import Final, Iterable

import numpy as np

from moptipy.spaces.intspace import IntSpace
from moptipy.utils.logger import CSV_SEPARATOR, KeyValueLogSection
from moptipy.utils.nputils import array_to_str
from moptipy.utils.types import type_error

#: the different choices
KEY_CHOICES: Final[str] = "choices"


class OrderedChoices(IntSpace):
    """Permutations of selections, stored as :class:`numpy.ndarray`."""

    def __init__(self, selections: Iterable[Iterable[int]]) -> None:
        """
        Create the space of permutations of selections.

        :param selections: an iterable of selections
        :raises TypeError: if one of the parameters has the wrong type
        :raises ValueError: if the parameters have the wrong value
        """
        if not isinstance(selections, Iterable):
            raise type_error(selections, "selections", Iterable)

        sets: Final[dict[int, tuple[int, ...]]] = {}

        min_v: int = 0
        max_v: int = 0
        total: int = 0
        counts: Counter[tuple[int, ...]] = Counter()
        blueprint: Final[list[int]] = []

        for i, sel in enumerate(selections):
            total += 1
            if not isinstance(sel, Iterable):
                raise type_error(sel, f"selections[{i}]", Iterable)
            sel_lst = tuple(sorted(sel))
            len_sel = len(sel_lst)
            if len_sel <= 0:
                raise ValueError(f"empty selection at index {i}: {sel}.")
            sel_set = set(sel_lst)
            if len(sel_set) != len_sel:
                raise ValueError(f"invalid selection {sel} at index {i} "
                                 f"contains duplicate element")
            blueprint.append(sel_lst[0])
            # build the selection map
            for e in sel_lst:
                if not isinstance(e, int):
                    raise type_error(e, f"selections[{i}]={sel}", int)
                if e in sets:
                    lst_found = sets[e]
                    if lst_found != sel_lst:
                        raise ValueError(
                            f"value {e} appears both in selection {sel_lst} "
                            f"(at permutation index {i}) and in selection "
                            f"{lst_found}!")
                    sel_lst = lst_found
                    continue  # if any sets[e] != sel_lst, we get error anyway
                sets[e] = sel_lst  # remember value
            counts[sel_lst] += 1

            # update the value range
            if total <= 1:
                min_v = sel_lst[0]
                max_v = sel_lst[-1]
            else:
                min_v = min(min_v, sel_lst[0])
                max_v = max(max_v, sel_lst[-1])

        if total <= 0:
            raise ValueError(
                "there must be at least one selection, "
                f"but got {selections}.")

        # creating super class
        super().__init__(total, min_v, max_v)

        #: the selector map
        self.choices: Final[dict[int, tuple[int, ...]]] = sets
        #: how often does each element choice appear?
        self.__counts: Final[dict[tuple[int, ...], int]] = {
            k: counts[k] for k in sorted(counts.keys())}
        blueprint.sort()

        #: The blueprint array, i.e., an ordered array holding the smallest
        #: value possible for each choice.
        self.blueprint: Final[np.ndarray] = np.array(
            blueprint, dtype=self.dtype)

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of this space to the given logger.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)

        strings: Final[list[str]] = []
        for sel, count in self.__counts.items():
            s = CSV_SEPARATOR.join(map(str, sel))
            strings.append(s if count <= 1 else (s + "*" + str(count)))
        logger.key_value(KEY_CHOICES, "/".join(strings))

    def create(self) -> np.ndarray:
        r"""
        Create an ordered choice, equal to :attr:`~OrderedChoices.blueprint`.

        :return: the ordered choices, an instance of :class:`numpy.ndarray`.

        >>> perm = OrderedChoices([[1, 3], [2, 4], [1, 3], [7, 5]])
        >>> x = perm.create()
        >>> print(perm.to_str(x))
        1;1;2;5
        """
        return self.blueprint.copy()  # Create copy of the blueprint.

    def validate(self, x: np.ndarray) -> None:
        """
        Validate a permutation of selections.

        :param x: the integer string
        :raises TypeError: if the string is not an :class:`numpy.ndarray`.
        :raises ValueError: if the element is not a valid permutation of the
            given choices
        """
        super().validate(x)

        usage: Final[Counter[tuple[int, ...]]] = Counter()
        dct: Final[dict[int, tuple[int, ...]]] = self.choices
        for j, i in enumerate(x):
            if i not in dct:
                raise ValueError(
                    f"invalid element {i} encountered at index"
                    f" {j} of string {array_to_str(x)}.")
            usage[dct[i]] += 1
        counts: Final[dict[tuple[int, ...], int]] = self.__counts
        for tup, cnt in usage.items():
            expected = counts[tup]
            if expected != cnt:
                raise ValueError(
                    f"encountered value from {tup} exactly {cnt} times "
                    f"instead of the expected {expected} times in {x}")

    def n_points(self) -> int:
        """
        Get the number of possible different permutations of the choices.

        :return: [factorial(dimension) / Product(factorial(count(e))
            for all e)] * Product(len(e)*count(e) for all e)
        """
        # get the basic permutation numbers now multiply them with the choices
        size = factorial(self.dimension)
        for lst, cnt in self.__counts.items():
            size = size // factorial(cnt)
            size = size * (len(lst) ** cnt)
        return size

    def __str__(self) -> str:
        """
        Get the name of this space.

        :return: "selperm{n}", where {n} is the length
        """
        return f"selperm{len(self.blueprint)}"

    def is_compatible_with_permutations(self) -> bool:
        """
        Check whether for compatibility with permutations with repetitions.

        Or, in other words, check whether the operators in package
        :mod:`~moptipy.operators.permutations` can safely be applied for
        elements of this space here.

        Permutations with repetitions are permutations where each element
        occurs exactly a given number of times. Our implementation of this
        space (:mod:`~moptipy.spaces.permutations`) ensures that there are
        at least two different elements. The unary and binary search
        operators defined in package :mod:`~moptipy.operators.permutations`
        rely on this fact. While these operators cannot explore the depth
        of the permutations-of-selections space here, they can be "compatible"
        to this space: Any element of the permutation-of-selections space is,
        by definition, a permutation with repetitions, as it contains one
        concrete manifestation per choice. Applying, for instance, a
        :mod:`~moptipy.operators.permutations.op1_swap2` operation to it,
        which swaps two different elements, still yields a valid and different
        permutation-of-selections. However, since the operators in
        :mod:`~moptipy.operators.permutations` always enforce that the
        resulting point is different from their input and *only* permute the
        elements, this can only work if we have at least two disjoint choices
        in our space definition. The function here checks this.

        :returns: `True` if and only if the operators in package
            :mod:`~moptipy.operators.permutations` can safely be applied to
            elements of this space
        """
        return len(self.__counts) > 1

    @staticmethod
    def signed_permutations(n: int) -> "OrderedChoices":
        """
        Create a space for signed permutations with values `1..n`.

        You would be much better off using
        :mod:`~moptipy.spaces.signed_permutations` instead of this space for
        signed permutations, though.

        :param n: the range of the values
        :returns: the permutations space

        >>> perm = OrderedChoices.signed_permutations(3)
        >>> perm.validate(perm.blueprint)
        >>> print(perm.blueprint)
        [-3 -2 -1]
        >>> print(perm.n_points())  # 3! * (2 ** 3) = 6 * 8 = 48
        48
        >>> perm = OrderedChoices.signed_permutations(4)
        >>> perm.validate(perm.blueprint)
        >>> print(perm.blueprint)
        [-4 -3 -2 -1]
        >>> print(perm.n_points())  # 4! * (2 ** 4) = 24 * 16 = 384
        384
        """
        return OrderedChoices([-i, i] for i in range(1, n + 1))
