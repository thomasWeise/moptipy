"""An archive of solutions to a multi-objective problems."""

from typing import Any, Final

import numpy as np

from moptipy.api.component import Component
from moptipy.api.mo_utils import lexicographic
from moptipy.utils.nputils import array_to_str
from moptipy.utils.types import type_error


class MORecord:
    """
    A record for the multi-objective archive.

    The default sorting order of multi-objective records is lexicographic
    based on the objective value vector.

    >>> import numpy as npx
    >>> mr1 = MORecord("xxx", npx.array([1, 2, 3], int))
    >>> print(mr1.x)
    xxx
    >>> print(mr1.fs)
    [1 2 3]
    >>> print(mr1)
    fs=1;2;3, x=xxx
    >>> mr2 = MORecord("yyy", npx.array([1, 2, 1], int))
    >>> print(mr2.x)
    yyy
    >>> print(mr2.fs)
    [1 2 1]
    >>> mr1 < mr2
    False
    >>> mr2 < mr1
    True
    """

    def __init__(self, x: Any, fs: np.ndarray) -> None:
        """
        Create a multi-objective record.

        :param x: the point in the search space
        :param fs: the vector of objective values
        """
        if x is None:
            raise TypeError("x must not be None")
        #: the point in the search space
        self.x: Final[Any] = x
        if not isinstance(fs, np.ndarray):
            raise type_error(fs, "fs", np.ndarray)
        #: the vector of objective values
        self.fs: Final[np.ndarray] = fs

    def __lt__(self, other) -> bool:
        """
        Compare for sorting.

        :param other: the other record

        >>> import numpy as npx
        >>> r1 = MORecord("a", npx.array([1, 1, 1]))
        >>> r2 = MORecord("b", npx.array([1, 1, 1]))
        >>> r1 < r2
        False
        >>> r2 < r1
        False
        >>> r2 = MORecord("b", npx.array([1, 1, 2]))
        >>> r1 < r2
        True
        >>> r2 < r1
        False
        >>> r1 = MORecord("a", npx.array([2, 1, 1]))
        >>> r1 < r2
        False
        >>> r2 < r1
        True
        """
        return lexicographic(self.fs, other.fs) < 0

    def __str__(self):
        """
        Get the string representation of this record.

        :returns: the string representation of this record

        >>> import numpy as npx
        >>> r = MORecord(4, npx.array([1, 2, 3]))
        >>> print(r)
        fs=1;2;3, x=4
        """
        return f"fs={array_to_str(self.fs)}, x={self.x}"


class MOArchivePruner(Component):
    """A strategy for pruning an archive of solutions."""

    def prune(self, archive: list[MORecord], n_keep: int, size: int) -> None:
        """
        Prune an archive.

        After invoking this method, the first `n_keep` entries in `archive`
        are selected to be preserved. The remaining entries
        (at indices `n_keep...len(archive)-1`) can be deleted.

        Pruning therefore is basically just a method of sorting the archive
        according to a preference order of solutions. It will not delete any
        element from the list. The caller can do that afterwards if she wants.

        This base class just provides a simple FIFO scheme.

        :param archive: the archive, i.e., a list of tuples of solutions and
            their objective vectors
        :param n_keep: the number of solutions to keep
        :param size: the current size of the archive
        """
        if size > n_keep:
            n_delete: Final[int] = size - n_keep
            move_to_end: Final[list[MORecord]] = archive[:n_delete]
            archive[0:n_keep] = archive[n_delete:size]
            archive[size - n_delete:size] = move_to_end

    def __str__(self):
        """
        Get the name of this archive pruning strategy.

        :returns: the name of this archive pruning strategy
        """
        return "fifo"


def check_mo_archive_pruner(pruner: Any) -> MOArchivePruner:
    """
    Check whether an object is a valid instance of :class:`MOArchivePruner`.

    :param pruner: the multi-objective archive pruner
    :return: the object
    :raises TypeError: if `pruner` is not an instance of
        :class:`MOArchivePruner`

    >>> check_mo_archive_pruner(MOArchivePruner())
    fifo
    >>> try:
    ...     check_mo_archive_pruner('A')
    ... except TypeError as te:
    ...     print(te)
    pruner should be an instance of moptipy.api.mo_archive.\
MOArchivePruner but is str, namely 'A'.
    >>> try:
    ...     check_mo_archive_pruner(None)
    ... except TypeError as te:
    ...     print(te)
    pruner should be an instance of moptipy.api.mo_archive.\
MOArchivePruner but is None.
    """
    if isinstance(pruner, MOArchivePruner):
        return pruner
    raise type_error(pruner, "pruner", MOArchivePruner)
