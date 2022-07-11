"""A multi-objective archive pruner based on distance."""

from collections import Counter
from math import inf
from typing import List, Final, Set, Iterable, Optional, Union

import numpy as np

from moptipy.api.mo_archive_pruner import MOArchivePruner
from moptipy.api.mo_utils import MOArchive
from moptipy.utils.nputils import np_number_to_py_number
from moptipy.utils.types import type_error


class KeepFarthest(MOArchivePruner):
    """
    Keep the best records of selected dimensions and those farthest from them.

    This archive pruner proceeds in two steps:

    First, it keeps the records that minimize a user-provided set of
    dimensions.

    Second, it iteratively adds the records to the selection which have the
    maximum minimal distance to the already selected ones in terms of the
    square distance over the normalized dimensions.

    This should provide some sort of diversity in the archive while allowing
    the user to preserve optima in certain dimensions.
    """

    def __init__(self, dimension: int,
                 keep_best_of_dimension: Optional[Iterable[int]] = None):
        """
        Create the distance-based pruner.

        :param dimension: the dimension of the problem
        :param keep_best_of_dimension: the dimensions of which we will always
            keep the best record, `None` (=default) for all dimensions
        """
        super().__init__()

        if not isinstance(dimension, int):
            raise type_error(dimension, "dimension", int)
        if dimension <= 0:
            raise ValueError(f"dimension={dimension} is not allowed")
        if keep_best_of_dimension is None:
            keep_best_of_dimension = range(dimension)
        if not isinstance(keep_best_of_dimension, Iterable):
            raise type_error(keep_best_of_dimension,
                             "keep_best_of_dimension", Iterable)

        #: the array for minima
        self.__min: Final[List[Union[int, float, np.number]]] = \
            [inf] * dimension
        #: the array for maxima
        self.__max: Final[List[Union[int, float, np.number]]] = \
            [-inf] * dimension
        #: the list of items to preserve per dimension
        self.__preserve: Final[List[Optional[Set]]] = [None] * dimension
        for d in keep_best_of_dimension:
            self.__preserve[d] = set()
        #: the list of all items to preserve
        self.__all_preserve: Final[List[Set]] = []
        for p in self.__preserve:
            if p is not None:
                self.__all_preserve.append(p)
        if len(self.__all_preserve) <= 0:
            raise ValueError(
                f"there must be at least one dimension of which we keep the "
                f"best, but keep_best_of_dimension={keep_best_of_dimension}!")
        #: the counter for keeping the best
        self.__counter: Final[Counter] = Counter()
        #: the chosen elements to keep
        self.__chosen: Final[List[int]] = []
        #: the minimal distances
        self.__min_dists: Final[List[float]] = []

    def prune(self, archive: MOArchive, n_keep: int) -> None:
        """
        Preserve the best of certain dimensions and keep the rest diverse.

        :param archive: the archive, i.e., a list of tuples of solutions and
            their objective vectors
        :param n_keep: the number of solutions to keep
        """
        count: Final[int] = len(archive)
        if count <= n_keep:
            return

        # set up basic variables
        mi: Final[List[Union[int, float, np.number]]] = self.__min
        ma: Final[List[Union[int, float, np.number]]] = self.__max
        dim: Final[int] = len(mi)
        preserve: Final[List[Optional[Set]]] = self.__preserve
        all_preserve: Final[List[Set]] = self.__all_preserve
        for i in range(dim):
            mi[i] = -inf
            ma[i] = inf
        for p in all_preserve:
            p.clear()
        counter: Final[Counter] = self.__counter
        counter.clear()
        chosen: Final[List[int]] = self.__chosen
        chosen.clear()
        min_dists: Final[List[float]] = self.__min_dists
        mdl: int = len(min_dists)
        for i in range(min(mdl, count)):
            min_dists[i] = inf
        if mdl < count:
            min_dists.extend([inf] * count)

        # get the ranges of the dimension and remember the record with
        # the minimal value per dimension
        for idx, ind in enumerate(archive):
            fs = ind[0]
            for i, f in enumerate(fs):
                if f <= mi[i]:
                    q: Optional[Set[int]] = preserve[i]
                    if f < mi[i]:
                        mi[i] = f
                        if q is not None:
                            q.clear()
                    if q is not None:
                        q.add(idx)
                if f > ma[i]:
                    ma[i] = f

        # the number of selected records be 0 at the beginning
        selected: int = 0

        # In a first step, we want to keep the minimal elements of the
        # selected objectives. Now there might be non-dominated records that
        # are minimal in more than one objective. In this case, we prefer
        # those that can satisfy several objectives. So we first check how
        # many objectives are minimized by each minimal element. We then
        # pick the one that satisfies most. We only pick one minimal element
        # per objective.

        # Count how often the records were selected for minimizing an objective
        for p in all_preserve:
            counter.update(p)

        # Now pick the elements that minimize most objectives.
        for maxc in sorted(sorted(counter.keys()),  # sort keys for stability
                           key=lambda kk: -counter[kk]):
            found: bool = False
            for p in all_preserve:
                if maxc in p:     # If the objective can be minimized by the
                    p.clear()     # element, we don't need to minimize it with
                    found = True  # another element and can keep this one.
            if found:
                chosen.append(maxc)
        chosen.sort()  # Sort by index: We swap the selected records forward

        # Now preserve the selected records by moving them to the front.
        for choseni in chosen:
            archive.insert(selected, archive.pop(choseni))
            selected = selected + 1
            if selected >= n_keep:
                return

        # Now we have the elements that minimize the selected dimensions.
        # Now we prepare the distances to ensure that we do not get any
        # overflow when normalizing them.
        for i in range(dim):
            mi[i] = mii = np_number_to_py_number(mi[i])
            maa = np_number_to_py_number(ma[i]) - mii
            ma[i] = maa if maa > 0 else 1.0  # ensure finite even on 0 ranges

            # Now we fill up the archive with those records most different from
        # the already included ones based on the square distance in the
        # normalized dimensions. In each iteration, we compute the minimal
        # normalized distance of each element to the already-selected ones.
        # We then keep the one which has the largest minimal distance and add
        # it to the selection.
        dist_update_start: int = 0
        while selected < n_keep:  # until we have selected sufficiently many
            max_dist: float = -inf  # the maximum distance to the selected
            max_dist_idx: int = selected  # the index of that record
            for rec_idx in range(selected, count):  # iterate over unselected
                min_dist_rec: float = min_dists[rec_idx]  # min dist so far
                rec: np.ndarray = archive[rec_idx][0]  # objective vector
                for cmp_idx in range(dist_update_start, selected):
                    cmp: np.ndarray = archive[cmp_idx][0]  # objective vector
                    ds: float = 0.0  # normalized square distance
                    for dd in range(dim):  # compute normalized distance
                        ddd: float = float(cmp[dd] - mi[dd]) / ma[dd] \
                            - float(rec[dd] - mi[dd]) / ma[dd]
                        ds = ds + float(ddd * ddd)
                    if ds < min_dist_rec:  # is this one closer?
                        min_dist_rec = ds  # remember
                min_dists[rec_idx] = min_dist_rec  # remember
                if min_dist_rec > max_dist:  # keep record with largest
                    max_dist = min_dist_rec  # normalized distance
                    max_dist_idx = rec_idx   # remember its index
            # swap the record to the front of the archive
            archive.insert(selected, archive.pop(max_dist_idx))
            del min_dists[max_dist_idx]
            min_dists.insert(selected, inf)
            dist_update_start = selected
            selected = selected + 1

    def __str__(self):
        """
        Get the name of this pruning strategy.

        :returns: always `"keepFarthest"`
        """
        return "keepFarthest"
