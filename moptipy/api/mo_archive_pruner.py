"""An archive of solutions to a multi-objective problems."""

from typing import Final

from moptipy.api.component import Component
from moptipy.api.mo_utils import MOArchive
from moptipy.utils.types import type_error


class MOArchivePruner(Component):
    """A strategy for pruning an archive of solutions."""

    def prune(self, archive: MOArchive, n_keep: int) -> None:
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
        """
        end: Final[int] = len(archive)
        if end > n_keep:
            n_delete: Final[int] = end - n_keep
            move_to_end: Final[MOArchive] = archive[:n_delete]
            archive[0:n_keep] = archive[n_delete:end]
            archive[end - n_delete:end] = move_to_end

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
