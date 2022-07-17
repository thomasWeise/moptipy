"""A multi-objective archive pruner based on distance."""

from collections import Counter
from math import inf
from typing import List, Final, Set, Iterable, Optional, cast

import numpy as np

from moptipy.api.mo_archive import MOArchivePruner, MORecord
from moptipy.api.mo_problem import MOProblem, check_mo_problem
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.nputils import DEFAULT_FLOAT, DEFAULT_INT, \
    DEFAULT_UNSIGNED_INT, KEY_NUMPY_TYPE, val_numpy_type
from moptipy.utils.types import type_error

#: the numpy data type for min, max, and dist arrays
KEY_NUMPY_TYPE_COMPUTE: Final[str] = f"{KEY_NUMPY_TYPE}Cmp"


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

    def __init__(self, problem: MOProblem,
                 keep_best_of_dimension: Optional[Iterable[int]] = None):
        """
        Create the distance-based pruner.

        :param problem: the multi-objective optimization problem
        :param keep_best_of_dimension: the dimensions of which we will always
            keep the best record, `None` (=default) for all dimensions
        """
        super().__init__()

        check_mo_problem(problem)
        dimension: Final[int] = problem.f_dimension()
        if not isinstance(dimension, int):
            raise type_error(dimension, "problem.f_dimension()", int)
        if dimension <= 0:
            raise ValueError(f"dimension={dimension} is not allowed")
        if keep_best_of_dimension is None:
            keep_best_of_dimension = range(dimension)
        if not isinstance(keep_best_of_dimension, Iterable):
            raise type_error(keep_best_of_dimension,
                             "keep_best_of_dimension", Iterable)

        tempdt: np.dtype = problem.f_create().dtype
        minmax_dtype: np.dtype = DEFAULT_FLOAT
        pinf: np.number = cast(np.number, np.inf)
        ninf: np.number = cast(np.number, -np.inf)
        if tempdt.kind in ('i', 'u'):
            if tempdt.kind == 'u':
                minmax_dtype = DEFAULT_UNSIGNED_INT
            else:
                minmax_dtype = DEFAULT_INT
            ii = np.iinfo(minmax_dtype)
            pinf = cast(np.number, ii.max)
            ninf = cast(np.number, ii.min)

        #: the array for minima
        self.__min: Final[np.ndarray] = np.empty(dimension, minmax_dtype)
        #: the array for maxima
        self.__max: Final[np.ndarray] = np.empty(dimension, minmax_dtype)
        #: the initial number for minimum searching
        self.__pinf: Final[np.number] = pinf
        #: the initial number for maximum searching
        self.__ninf: Final[np.number] = ninf
        #: the default divisor
        self.__div: Final[np.ndarray] = np.empty(dimension, minmax_dtype)
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
        self.__min_dists: np.ndarray = np.empty(8, DEFAULT_FLOAT)

    def prune(self, archive: List[MORecord], n_keep: int) -> None:
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
        mi: Final[np.ndarray] = self.__min
        mi.fill(self.__pinf)
        ma: Final[np.ndarray] = self.__max
        ma.fill(self.__ninf)
        div: Final[np.ndarray] = self.__div
        dim: Final[int] = len(mi)
        preserve: Final[List[Optional[Set]]] = self.__preserve
        all_preserve: Final[List[Set]] = self.__all_preserve
        for p in all_preserve:
            p.clear()
        counter: Final[Counter] = self.__counter
        counter.clear()
        chosen: Final[List[int]] = self.__chosen
        chosen.clear()
        min_dists: np.ndarray = self.__min_dists
        mdl: int = len(min_dists)
        if mdl < count:
            self.__min_dists = min_dists = np.full(count, inf, DEFAULT_FLOAT)
        else:
            min_dists.fill(inf)

        # get the ranges of the dimension and remember the record with
        # the minimal value per dimension
        for idx, ind in enumerate(archive):
            fs: np.ndarray = ind.fs
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
            maa = ma[i]
            if maa >= self.__pinf:
                raise ValueError(f"maximum of dimension {i} is {maa}")
            mii = mi[i]
            if mii <= self.__ninf:
                raise ValueError(f"minimum of dimension {i} is {mii}")
            if maa < mii:
                raise ValueError(
                    f"minimum of dimension {i} is {mii} and maximum is {maa}")
            dv = 1 if maa <= mii else maa - mii  # ensure finite on div=0
            if dv < 0:
                raise ValueError(f"{maa} - {mii} = {dv}?")
            div[i] = dv

        # Now we fill up the archive with those records most different from
        # the already included ones based on the square distance in the
        # normalized dimensions. In each iteration, we compute the minimal
        # normalized distance of each element to the already-selected ones.
        # We then keep the one which has the largest minimal distance and add
        # it to the selection.
        dist_update_start: int = 0
        while selected < n_keep:  # until we have selected sufficiently many
            max_dist: float = -inf  # the maximum distance to be selected
            max_dist_idx: int = selected  # the index of that record
            for rec_idx in range(selected, count):  # iterate over unselected
                min_dist_rec: float = min_dists[rec_idx]  # min dist so far
                rec: np.ndarray = archive[rec_idx].fs  # objective vector
                for cmp_idx in range(dist_update_start, selected):
                    cmp: np.ndarray = archive[cmp_idx].fs  # objective vector
                    dst: float = float(np.linalg.norm(
                        ((cmp - mi) / div) - ((rec - mi) / div)))
                    if dst < min_dist_rec:  # is this one closer?
                        min_dist_rec = dst  # remember
                min_dists[rec_idx] = min_dist_rec  # remember
                if min_dist_rec > max_dist:  # keep record with largest
                    max_dist = min_dist_rec  # normalized distance
                    max_dist_idx = rec_idx   # remember its index
            # swap the record to the front of the archive
            archive.insert(selected, archive.pop(max_dist_idx))
            min_dists[selected + 1:max_dist_idx + 1] \
                = min_dists[selected:max_dist_idx]
            min_dists[selected] = max_dist
            dist_update_start = selected
            selected = selected + 1

    def __str__(self):
        """
        Get the name of this pruning strategy.

        :returns: always `"keepFarthest"`
        """
        return "keepFarthest"

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """Log the parameters."""
        super().log_parameters_to(logger)
        logger.key_value(KEY_NUMPY_TYPE_COMPUTE,
                         val_numpy_type(self.__min.dtype))
