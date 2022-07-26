"""An archive of solutions to a multi-objective problems."""

from typing import Final, List, Any

import numpy as np

from moptipy.api.component import Component
from moptipy.api.mo_utils import lexicographic
from moptipy.utils.types import type_error


class MORecord:
    """
    A record for the multi-objective archive.

    The default sorting order of multi-objective records is lexicographic
    based on the objective value vector.
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

        >>> import numpy as np
        >>> r1 = MORecord("a", np.array([1, 1, 1]))
        >>> r2 = MORecord("b", np.array([1, 1, 1]))
        >>> r1 < r2
        False
        >>> r2 < r1
        False
        >>> r2 = MORecord("b", np.array([1, 1, 2]))
        >>> r1 < r2
        True
        >>> r2 < r1
        False
        >>> r1 = MORecord("a", np.array([2, 1, 1]))
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

        >>> import numpy as np
        >>> r = MORecord(4, np.array([1, 2, 3]))
        >>> print(r)
        fs=1;2;3, x=4
        """
        return f"fs={';'.join([str(a) for a in self.fs])}, x={self.x}"


class MOArchivePruner(Component):
    """A strategy for pruning an archive of solutions."""

    def prune(self, archive: List[MORecord], n_keep: int, size: int) -> None:
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
            move_to_end: Final[List[MORecord]] = archive[:n_delete]
            archive[0:n_keep] = archive[n_delete:size]
            archive[size - n_delete:size] = move_to_end

    def __str__(self):
        """
        Get the name of this archive pruning strategy.

        :returns: the name of this archive pruning strategy
        """
        return "fifo"


def check_mo_archive_pruner(pruner: MOArchivePruner) -> MOArchivePruner:
    """
    Check whether an object is a valid instance of :class:`MOArchivePruner`.

    :param pruner: the multi-objective archive pruner
    :return: the object
    :raises TypeError: if `pruner` is not an instance of
        :class:`MOArchivePruner`
    """
    if not isinstance(pruner, MOArchivePruner):
        raise type_error(pruner, "pruner", MOArchivePruner)
    return pruner
