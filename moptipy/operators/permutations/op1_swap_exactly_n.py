"""
An operator trying to change exactly `n` elements in a permutation.

This is an operator with a step size
(:class:`moptipy.api.operators.Op1WithStepSize`) and the step size determines
how many elements of a permutation should be changed.
If you are working on permutations without repetitions, the operator in
:mod:`~moptipy.operators.permutations.op1_swap_try_n` is the better and faster
choice. On normal permutations, it will be equivalent to
:mod:`~moptipy.operators.permutations.op1_swap_exactly_n`, but faster. On
Permutations with repetitions, it will still be faster but less precisely
enforces the number of swaps.

Let's say we have a "normal" permutation where each element occurs once.
The permutation has length `n`, let's say that its
:attr:`~moptipy.spaces.permutations.Permutations.blueprint` is
`[0, 1, 2, 3, 4, ..., n-1]`. Then, with one application of the operator, we
can change no less than two elements (`step_size=0.0`) and no more than `n`
(`step_size=1.0`). Clearly we cannot change only one element, because what
different value could we put there? We would violate the "permutation nature."
We can also not change more than `n` elements, obviously.

What our operator does is then it computes `m` by using
:func:`~moptipy.operators.tools.exponential_step_size`.
This ensures that `m=2` for `step_size=0.0` and `m` is maximal for
`step_size=1.0`. In between the two, it extrapolates exponentially. This means
that small values of `step_size` will lead to few swap moves, regardless of
the length of the permutation and many swaps are performed only for
`step_size` values close to `1`.
Then it will pick `m` random indices and permute the elements at them such
that none remains at its current position.
That is rather easy to do.
Unfortunately, :class:`~moptipy.spaces.permutations.Permutations` allows also
permutations with repetitions. Here, things get dodgy.

Let's say we have the blueprint `[0, 1, 1, 1, 1]`. Then we can only change
exactly two elements.

>>> print(get_max_changes([0, 1, 1, 1, 1]))
2

If we have the blueprint `[0, 0, 1, 1, 1]`, then we can change two elements.
We can also change four elements. But there is no way to change three elements!

>>> print(get_max_changes([0, 0, 1, 1, 1]))
4

So in the general case, we determine the maximum number `mx` of positions that
can be changed with one move via the function :func:`get_max_changes`.
We can still translate a `step_size` to a number `m` by doing
`m = 2 + int(round(step_size * (mx - 2)))`.
However, it is not clear whether all moves from `m=2` to `m=mx` are actually
possible.

And even if they are possible, they might be very hard to sample randomly.
Some moves touching a large number `m` of positions may be restricted to
swap the elements at very specific indicies in a very specific order. Finding
such a move by chance may be rather unlikely.

Indeed, a search operator should do random moves. I did not find a way to
construct moves according to a policy which is both random and can yield any
move.
So I divide the operator into two steps:

First, the function :func:`find_move` tries to find a move for a given `m`
in a random fashion. This function tries to find the sequence of indices for a
cyclic swap where each element is different from both its successor and
predecessor. It builds such index sequences iteratively. This may fail:
as in the example above, for some values of `m` it might just not be
possible to construct a suitable sequence, either because there is none or
because building it randomly has too low of a chance of success. Hence, the
function tries to build at most `max_trials` sequences. Whenever building a
sequence, it also remembers the longest-so-far cyclic move. If that one is
shorter than `m` but all `max_trials` trials are exhausted, it returns this
sequence instead. So in summary, this function tries to find the longest
cyclic swap which is not longer than `m` and returns it. It may be shorter
than `m`. If we deal with permutations where each value occurs only once, this
function is guaranteed to find a sequence of length `m` in the first trial,
i.e., it does not waste runtime.

Once a move was found, the function :func:`apply_move` applies it. Now, as
said, :func:`find_move` discovers cyclic changes that are reasonably random.
However, cyclic changes are not the only possible moves of length `m`. For
example, if we have the permutation blueprint `[0, 1, 2, 3, 4]`, a move of
length 4 could be to exchange `0` with `1` and to swap `2` with `3`.
:func:`find_move` cannot find this move, but it could find a cyclic swap of
`0` to `1`, `1` to `2`, `2` to `3`, and `3` to `1`. So it could find the
right indices for such a move, just restricted to the cyclic swapping. So
what :func:`apply_move` tries to do is to permute the indices discovered by
:func:`find_move` randomly and check whether this would still yield a feasible
move changing exactly `m` locations. Any shuffling of the elements at the
selected position which avoids putting the original values into the original
positions would do. Sometimes, most such shuffles work out, e.g., if we work
on the space of permutations where each element occurs once. Sometimes, the
only sequence that works is indeed the cyclic move and shuffling it cannot
work. So :func:`apply_move` again has a limit for the maximum number of
attempts to find a shuffle that works out. If it finds one, then it applies
it as move. If it does not find one and the trials are exhausted, it randomly
choses whether to apply the cyclic move as a cycle to the left or as a cycle
to the right. Either way, it will change exactly `m` positions of the
permutation, as prescribed by the move. Cycling one step in either direction
will always work, since each element is different from both its predecessor
and successor. (Cycling more than one step (but less than `m`) could
potentially fail in permutations with repetitions, because there is no
guarantee that any element is different from its successor's successor.)

Now the overall operator just plugs these two functions together. It also adds
one slight improvement: If we demand to change a number `q` of locations for
the first time and :func:`find_move` fails to find a move of length `q` but
instead offers one of length `p<q`, then the operator remembers this. The next
time we ask to change `q` positions, it will directly try to change only `p`.
This memory is reset in :meth:`~Op1SwapExactlyN.initialize`.

A similar but much more light-weight and faster operator is given in
:mod:`~moptipy.operators.permutations.op1_swap_try_n`. That operator also
tries to perform a given number of swaps, but puts in much less effort to
achieve this goal, i.e., it will only perform a single attempt.
"""
from typing import Counter, Final, Iterable

import numba  # type: ignore
import numpy as np
from numpy.random import Generator

from moptipy.api.operators import Op1WithStepSize
from moptipy.operators.tools import exponential_step_size
from moptipy.spaces.permutations import Permutations
from moptipy.utils.logger import CSV_SEPARATOR, KeyValueLogSection
from moptipy.utils.nputils import DEFAULT_INT, fill_in_canonical_permutation
from moptipy.utils.types import check_int_range, type_error


def get_max_changes(blueprint: Iterable[int]) -> int:
    """
    Get the maximum number of changes possible for a given permutation.

    :param blueprint: the blueprint permutation
    :returns: the maximum number of changes possible

    >>> get_max_changes("1")
    0
    >>> get_max_changes("11")
    0
    >>> get_max_changes("12")
    2
    >>> get_max_changes("123")
    3
    >>> get_max_changes("1233")
    4
    >>> get_max_changes("12333")
    4
    >>> get_max_changes("1234")
    4
    >>> get_max_changes("12344")
    5
    >>> get_max_changes("123444")
    6
    >>> get_max_changes("1234444")
    6
    >>> get_max_changes("12334444")
    8
    >>> get_max_changes("123344445")
    9
    >>> get_max_changes("1233444455")
    10
    >>> get_max_changes("12334444555")
    11
    >>> get_max_changes("123344445555")
    12
    >>> get_max_changes("1233444455555")
    13
    >>> get_max_changes("12334444555555")
    14
    >>> get_max_changes("123344445555555")
    15
    >>> get_max_changes("1233444455555555")
    16
    >>> get_max_changes("12334444555555555")
    16
    >>> get_max_changes("112233")
    6
    >>> get_max_changes("11223344")
    8
    >>> get_max_changes("112233445")
    9
    >>> get_max_changes("1122334455")
    10
    >>> get_max_changes("11223344555")
    11
    >>> get_max_changes("112233445555555")
    15
    >>> get_max_changes("1122334455555555")
    16
    >>> get_max_changes("11223344555555555")
    16
    """
    # Create tuples of (count, negative priority, value).
    counts: Final[list[list[int]]] = [
        [a[1], 0, a[0]] for a in Counter[int](blueprint).most_common()]
    if len(counts) <= 1:
        return 0

# We simulate a chain swap. We begin with the element that appears the least
# often. We take this element and reduce its count. Next, we take the element
# that appears the most often. We reduce its count. And so on. Ties are broken
# by prioritizing the element that was not used the longest. If the element
# that we want to pick is the same as the previously picked one, we skip over
# it. If no element can be picked, we quit.
    changes: int = 0
    last: int | None = None
    found: bool = True
    smallest: bool = True
    while found:
        found = False
# We sort alternatingly sometimes we pick the smallest, sometimes the largest.
        if smallest:
            counts.sort()
        else:
            counts.sort(key=lambda a: [-a[0], a[1]])
        smallest = not smallest  # realize the alternating sorting
# Now we iterate over the sorted list and pick the first suitable element.
# I know: There will be at most two iterations of this loop ... but whatever.
        for idx, chosen in enumerate(counts):
            current = chosen[2]
            if current == last:  # We cannot pick the same element twice
                continue         # in a row. So we skip over it.
            cnt = chosen[0]
            if cnt <= 1:  # How often does this element exist in the list?
                del counts[idx]  # No more elements of this type.
            else:
                chosen[0] = cnt - 1  # There is at least one more left over.
            changes = changes + 1  # Yeah, we can increase the changes.
            chosen[1] = changes  # Increase negative priority.
            found = True  # We found an element.
            last = current  # Remember the element.
            break  # Quit (after at most two loop iterations).
    if changes <= 1:  # This cannot be.
        raise ValueError(
            f"Error in counting possible changes for {blueprint}.")
    return changes


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def find_move(x: np.ndarray, indices: np.ndarray, step_size: int,
              random: Generator, max_trials: int,
              temp: np.ndarray) -> np.ndarray:
    """
    Find a move of at most step_size changes and store it in indices.

    For any pure permutation `x`, we can basically find any move between
    `step_size=2` and `len(x)`. If we have permutations with repetitions, the
    upper limit becomes smaller, because it may not be possible to change all
    elements at once. For example, in `[0, 0, 1, 1, 1, 1, 1]`, we can change
    at most four elements. For this purpose, :func:`get_max_changes` exists.
    It tells us the upper limit of changes, which would be four here:

    >>> print(get_max_changes([0, 0, 1, 1, 1, 1, 1]))
    4

    Yet, this does not necessarily mean that we can make all moves including
    2, 3, and 4 changes. If you look at the above example, you may find that
    it is impossible to change exactly 3 locations. We can only actually do
    step sizes of 2 and 4, but neither 3 nor 5, 6, nor 7.
    In reality, even if it was possible to make a big move, it might be very
    unlikely to be able to this in a *random* fashion. If you look at the
    code of :func:`get_max_changes`, it tries to find the longest possible
    cyclic move by working through the elements of the permutation in a very
    specific order. Using a different (or random) order would yield shorter
    moves.

    Our method :func:`find_move` here therefore tries to find the longest
    possible cyclic swap involving at most `step_size` indices. It may find
    such a move that uses exactly `step_size` indices. But it may also find
    a shorter move because either the perfect complete feasible move touching
    `step_size` indices does not exist or because finding it in a random
    fashion may take too long. (For this purpose, `max_trials` exists to
    limit the search for the right move.)

    For normal permutations, it will find the move of exactly the right
    length. For permutations with repetitions, it has a decent chance to find
    a move of the length `step_size` if a) such a move exists and
    b) `step_size` is not too big. Otherwise, it should find a shorter but
    still reasonably large move.

    :param x: the permutation to be modified in the end
    :param indices: the set of indices, which must be the canonical
        permutation of `0...len(x)-1`
    :param step_size: the step size, i.e., the number of elements to change
    :param random: the random number generator
    :param max_trials: the maximum number of trials
    :param temp: a temporary array
    :returns: the discovered move

    >>> import numpy as npx
    >>> from numpy.random import default_rng
    >>> gen = default_rng(12)
    >>> perm = npx.array([0, 1, 2, 3, 3, 3, 3, 3, 3], int)
    >>> use_indices = npx.array(range(len(perm)), int)
    >>> want_size = len(perm)
    >>> print(want_size)
    9
    >>> print(get_max_changes(perm))
    6
    >>> use_temp = npx.empty(len(perm), int)
    >>> res = find_move(perm, use_indices, want_size, gen, 1000, use_temp)
    >>> print(res)
    [5 2 8 1 4 0]
    >>> perm2 = perm.copy()
    >>> perm2[res] = perm2[npx.roll(res, 1)]
    >>> print(f"{perm} vs. {perm2}")
    [0 1 2 3 3 3 3 3 3] vs. [3 3 3 3 1 0 3 3 2]
    >>> print(sum(perm != perm2))
    6
    >>> perm = npx.array([1, 3, 5, 9, 4, 2, 11, 7], int)
    >>> want_size = len(perm)
    >>> print(want_size)
    8
    >>> use_indices = npx.array(range(len(perm)), int)
    >>> use_temp = npx.empty(len(perm), int)
    >>> res = find_move(perm, use_indices, want_size, gen, 1, use_temp)
    >>> print(res)
    [4 0 5 3 1 7 2 6]
    >>> print(len(res))
    8
    >>> perm2 = perm.copy()
    >>> perm2[res] = perm2[npx.roll(res, 1)]
    >>> print(f"{perm} vs. {perm2}")
    [ 1  3  5  9  4  2 11  7] vs. [ 4  9  7  2 11  1  5  3]
    >>> print(sum(perm != perm2))
    8
    """
    length: Final[int] = len(x)
    lm1: Final[int] = length - 1
    sm1: int = step_size - 1
    best_size: int = -1

# We spent at most `max_trials` trials to find a perfect move. The reason is
# that some values of `step_size` might be impossible to achieve, other may
# be achievable only with a very low probability. So we need a trial limit to
# avoid entering a very long or even endless loop.
    for _ in range(max_trials):
        current_best_size: int = -1  # longest feasible move this round
# We aim to store a suitable selection of indices at indices[0:step_size].
# Like in :mod:`moptipy.operators.permutations.op1_swapn`, we try to create a
# feasible move as a series of cyclic swaps. Basically, the element at index
# i1 would be moved to index i2, the element at i2 to i3, ..., the element at
# the last index ix to i1. We make sure that all indices are used at most
# once.
# We proceed similar to a partial Fisher-Yates shuffle.
# The variable "found" is the number of indices that we have fixed so far.
# For each new position, we test the remaining indices in random order.
# An index can only be used if it points to an element different to what the
# last index points to, or else the cyclic swap would be meaningless.
# For the last index to be chosen, that element must also be different from
# the first element picked.
# So a potential index may not work. We test each index at most once by moving
# the tested indices into the range indices[found:tested] and keep increasing
# tested and reset it once we found a new acceptable index.
# This procedure, however, may end up in a dead end. We therefore need to wrap
# it into a loop that tries until success...  ...but we cannot do that because
# success may not be possible. Some moves may be impossible to do.
        i_idx: int = random.integers(0, length)  # Get the first random index.
        i_src: int = indices[i_idx]  # Pick the actual index at that index.
        indices[0], indices[i_idx] = i_src, indices[0]  # Swap it to 0.
        found: int = 1  # the number of found indices
        tested: int = 1  # the number of tested indices
        last = first = x[i_src]  # Get the value at the first index.
        continue_after: bool = found < sm1  # continue until step_size-1
        while True:
            i_idx = random.integers(tested, length)  # Get random index.
            i_dst: int = indices[i_idx]  # Pick the actual index.
            current: int = x[i_dst]  # Get value at the new index.
            can_end: bool = current != first
            accept: bool = (current != last) and (continue_after or can_end)
            if accept:
                # Now we move the new index into the "found" section.
                indices[found], indices[i_idx] = i_dst, indices[found]
                tested = found = found + 1  # found++, reset tested.
                last = current  # Store value from i_dst in last.
                if can_end:
                    current_best_size = found
                    if not continue_after:
                        return indices[0:found]  # We won!
                continue_after = found < sm1  # continue until step_size-1
                continue
            if tested >= lm1:  # Are we in a dead end?
                break  # and try again freshly.
            indices[tested], indices[i_idx] = i_dst, indices[tested]
            tested = tested + 1  # We have tested another value.

# If we get here, we did not succeed in creating a complete move this round.
# However, current_best_size should indicate one possible shorter move.
# It must be at least >= 2, because we will definitely be able to find two
# different elements in x. We will remember the longest feasible move
# discovered in all max_trials iterations.
        if current_best_size > best_size:
            best_size = current_best_size
            temp[0:best_size] = indices[0:best_size]

# If get here, we have completed the main loop. We could not find a complete
# feasible move that changes exactly step_size locations. However, we should
# have a shorter move by now stored in temp. So we copy its indices.
    return temp[0:best_size]


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def apply_move(x: np.ndarray, dest: np.ndarray, move: np.ndarray,
               random: Generator, max_trials: int) -> None:
    """
    Apply a given move copying from source `x` to `dest`.

    `move` must be the return value of :func:`find_move`.

    :param x: the source permutation
    :param dest: the destination permutation
    :param move: the move of length `step_size`
    :param random: the random number generator
    :param max_trials: a maximum number of trials

    >>> import numpy as npx
    >>> from numpy.random import default_rng
    >>> rand = default_rng(12)
    >>> xx = npx.array([0, 1, 2, 3, 3, 3, 3, 3, 3], int)
    >>> dd = npx.empty(len(xx), int)
    >>> mv = npx.array([5, 2, 8, 1, 4, 0], int)
    >>> print(len(mv))
    6
    >>> apply_move(xx, dd, mv, rand, 0)
    >>> print(dd)
    [3 3 3 3 0 2 3 3 1]
    >>> print(sum(dd != xx))
    6
    >>> apply_move(xx, dd, mv, rand, 1000)
    >>> print(dd)
    [3 3 3 3 2 1 3 3 0]
    >>> print(sum(dd != xx))
    6
    >>> xx = npx.array([1, 3, 5, 9, 4, 2, 11, 7], int)
    >>> dd = npx.empty(len(xx), int)
    >>> mv = npx.array([4, 0, 5, 3, 1, 7, 2, 6], int)
    >>> print(len(mv))
    8
    >>> apply_move(xx, dd, mv, rand, 0)
    >>> print(dd)
    [ 4  9  7  2 11  1  5  3]
    >>> print(sum(dd != xx))
    8
    >>> xx = npx.array([1, 3, 5, 9, 4, 2, 11, 7], int)
    >>> apply_move(xx, dd, mv, rand, 10)
    >>> print(dd)
    [ 5  4  2 11  7  3  9  1]
    >>> print(sum(dd != xx))
    8
    """
    dest[:] = x[:]  # First, we copy `x` to `dest`.
    step_size: Final[int] = len(move)  # Get the move size.
# We now try to replace the sub-string x[move] with a random permutation of
# itself. Since `move` is the return value of `find_move`, we know that it
# could be realized as a cyclic swap. However, we are not yet sure whether it
# can also be some other form of move, maybe swapping a few pairs of elements.
# We try to find this out in this loop by testing `max_trials` random
# permutations of the original subsequence.
    orig: np.ndarray = dest[move]   # Get the original subsequence.
    perm: np.ndarray = orig.copy()
    for _ in range(max_trials):  # Try to permute it.
        random.shuffle(perm)  # Shuffle the permutation
        if sum(perm != orig) == step_size:  # If all elements are now...
            dest[move] = perm  # ...at different places, accept the move and
            return  # quit
# If we get here, trying random permutations of the sequence did not work.
# We now simply perform a cyclic swap. A cyclic swap has the property that
# each element is different from both its predecessor and its successor.
# Hence, we could cycle left or cycle right. Now one could argue that this
# should not matter, since we have constructed the move in a random way via
# find_move. However, I am not sure. If we have a permutation with repetition,
# then finding certain move patterns could be more likely than others. For
# example, if one element occurs much more often than another one, one of its
# instance could be more likely chosen as first element of the move. A more
# rarely occurring element could then appear later in the move when there are
# not many other choices left. Hence, it might be that the moves found by
# find_move are not uniformly distributed over the realm of possible moves of
# a given step length. In this case, it may be useful to randomly choose
# whether to cycle left or right. We cannot cycle more than one step, though,
# because there is guarantee that an element is different from its successor's
# successor.
    dest[move] = np.roll(orig, 1 if random.integers(0, 2) == 0 else -1)


class Op1SwapExactlyN(Op1WithStepSize):
    """A unary search operator that swaps `n` (different) elements."""

    def __init__(self, perm: Permutations,
                 max_move_trials: int = 1000,
                 max_apply_trials: int = 1000) -> None:
        """
        Initialize the operator.

        :param perm: the base permutation
        :param max_move_trials: the maximum number of attempts to generate a
            fitting move
        :param max_apply_trials: the maximum number of attempts to apply a
            move via a random permutation
        """
        super().__init__()
        if not isinstance(perm, Permutations):
            raise type_error(perm, "perm", Permutations)

# If each value appears exactly once in the permutation, then we can
# easily do perm.dimension changes. If the permutation is with
# repetitions, then it might be fewer, so we need to check.
        is_pure_perm: Final[bool] = perm.n() == perm.dimension
        #: the maximum number of possible changes
        self.max_changes: Final[int] = check_int_range(
            perm.dimension if is_pure_perm
            else get_max_changes(perm.blueprint),
            "max_changes", 2, 100_000_000)
        #: the maximum number of attempts to find a move with the exact step
        #: size
        self.max_move_trials: Final[int] =\
            check_int_range(max_move_trials, "max_move_trials", 1)
        #: the maximum number of attempts to apply a random permutation of the
        #: move before giving up and applying it as cyclic swap
        self.max_apply_trials: Final[int] =\
            check_int_range(max_apply_trials, "max_apply_trials", 1)
        #: the set of chosen indices
        self.__indices: Final[np.ndarray] = np.empty(
            perm.dimension, DEFAULT_INT)
        #: a temporary array
        self.__temp: Final[np.ndarray] = np.empty(
            perm.dimension, DEFAULT_INT)
        #: the initial move map that maps step sizes to feasible sizes
        self.__initial_move_map: Final[dict[int, int]] = {i: i for i in range(
            2, perm.dimension + 1)} if is_pure_perm else {2: 2}
        #: the move map that maps step sizes to feasible sizes
        self.__move_map: Final[dict[int, int]] = {}

    def initialize(self) -> None:
        """Initialize this operator."""
        super().initialize()
        fill_in_canonical_permutation(self.__indices)
        self.__move_map.clear()
        self.__move_map.update(self.__initial_move_map)

    def op1(self, random: Generator, dest: np.ndarray, x: np.ndarray,
            step_size: float = 0.0) -> None:
        """
        Copy `x` into `dest` and then swap several different values.

        :param random: the random number generator
        :param dest: the array to receive the modified copy of `x`
        :param x: the existing point in the search space
        :param step_size: the number of elements to swap
        """
# compute the real step size based on the maximum changes
        use_step_size: int = exponential_step_size(
            step_size, 2, self.max_changes)
# look up in move map: Do we already know whether this step size works or can
# it be replaced with a similar smaller one?
        move_map: Final[dict[int, int]] = self.__move_map
        mapped_step_size = move_map.get(use_step_size, -1)
        if mapped_step_size >= 2:
            use_step_size = mapped_step_size

        move: Final[np.ndarray] = find_move(
            x, self.__indices, use_step_size, random, self.max_move_trials,
            self.__temp)

# If we did not yet know the move size, then update it.
        if mapped_step_size == -1:
            self.__move_map[use_step_size] = len(move)
        self.__move_map[use_step_size] = use_step_size
# Finally, apply the move.
        apply_move(x, dest, move, random, self.max_move_trials)

    def __str__(self) -> str:
        """
        Get the name of this unary operator.

        :returns: "swapxn", the name of this operator
        :retval "swapxn": always
        """
        return "swapxn"

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of this operator to the given logger.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        logger.key_value("maxChanges", self.max_changes)
        logger.key_value("maxMoveTrials", self.max_move_trials)
        logger.key_value("maxApplyTrials", self.max_apply_trials)

# Translate the move map to some strings like "2;5-8;12", meaning that all the
# moves 2, 5, 6, 7, 8, and 12 were performed and valid.
        kvs: list[int] = [k for k, v in self.__move_map.items() if k == v]
        kvs.sort()
        lk: Final[int] = len(kvs) - 1
        kvs = [(e if ((i <= 0) or (i >= lk) or kvs[i - 1] != (e - 1))
                or (kvs[i + 1] != (e + 1)) else -1)
               for i, e in enumerate(kvs)]
        kvs = [e for i, e in enumerate(kvs) if
               (i <= 0) or (i >= lk) or (e != -1) or (kvs[i - 1] != -1)]
        logger.key_value("moves", CSV_SEPARATOR.join(
            "-" if k < 0 else str(k) for k in kvs).replace(
            f"{CSV_SEPARATOR}-{CSV_SEPARATOR}", "-"))
