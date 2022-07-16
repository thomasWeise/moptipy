"""Functions for testing multi-objective archive pruners."""
from typing import Final, List, Iterable, Optional

import numpy as np
from numpy.random import Generator, default_rng

from moptipy.api.mo_archive import MOArchivePruner, \
    check_mo_archive_pruner, MORecordY
from moptipy.tests.component import validate_component
from moptipy.utils.types import type_error


def validate_mo_archive_pruner(pruner: MOArchivePruner,
                               dimensions: Iterable[int]) -> None:
    """
    Check whether an object is a moptipy multi-objective optimization pruner.

    This method checks whether the class is correct and whether the pruning
    follows the general contract: Interesting records in the list to be pruned
    are moved to the front, the ones to discard are moved to the back. No
    record is lost and none is duplicated.

    :param pruner: the multi-objective archive pruner to test
    :param dimensions: the dimensions to simulate
    :raises ValueError: if `mo_problem` is not a valid
        :class:`~moptipy.api.mo_problem.MOProblem`
    :raises TypeError: if values of the wrong types are encountered
    """
    if not isinstance(pruner, MOArchivePruner):
        raise type_error(pruner, "pruner", MOArchivePruner)
    check_mo_archive_pruner(pruner)
    validate_component(pruner)

    if not isinstance(dimensions, Iterable):
        raise type_error(dimensions, "dimensions", Iterable)

    random: Final[Generator] = default_rng()
    dtypes: Final[List[np.dtype]] = [np.dtype(x) for x in [
        np.int8, int, float]]
    mins: Final[List[int]] = [-128, -(1 << 61), -(1 << 51)]
    maxs: Final[List[int]] = [127, (1 << 61) - 1, (1 << 51)]
    tag: Final[str] = "bla"
    nothing: bool = True
    for dim in dimensions:
        if not isinstance(dim, int):
            raise type_error(dim, "dimensions[i]", int)
        nothing = False
        for i, dt in enumerate(dtypes):
            if not isinstance(dt, np.dtype):
                raise type_error(dt, "dt", np.dtype)
            alen = int(random.integers(2, 10))
            if not isinstance(alen, int):
                raise type_error(alen, "alen", int)
            amax = int(random.integers(1, alen + 1))
            if not isinstance(amax, int):
                raise type_error(amax, "amax", int)
            if not (0 < amax <= alen <= 10):
                raise ValueError(f"invalid amax={amax} and alen={alen}.")
            archive: List[MORecordY] = []
            archivec: List[MORecordY] = []

            for _ in range(alen):
                needed: bool = True
                rec: Optional[MORecordY] = None
                fs: Optional[np.ndarray] = None
                while needed:   # we make sure that all records are unique
                    fs = np.empty(dim, dt)
                    if not isinstance(fs, np.ndarray):
                        raise type_error(fs, "fs", np.ndarray)
                    if len(fs) != dim:
                        raise ValueError(f"len(fs)={len(fs)}!=dim={dim}")
                    for k in range(dim):
                        fs[k] = random.integers(mins[i], maxs[i])
                    rec = MORecordY(tag, fs, float(random.uniform(0, 1)))
                    if not isinstance(rec, MORecordY):
                        raise type_error(rec, "rec", MORecordY)
                    needed = False
                    for z in archive:
                        fs2 = z.fs
                        if np.array_equal(fs, fs2):
                            needed = True
                            break
                if (rec is None) or (fs is None):
                    raise ValueError("huh?")
                archive.append(rec)
                rec2 = MORecordY(tag, fs.copy(), rec.f)
                if (rec.x != rec2.x) \
                        or (not np.array_equal(rec2.fs, rec.fs)) \
                        or (rec2.f != rec.f):
                    raise ValueError(f"{rec} != {rec2}")
                archivec.append(rec2)

            # done creating archive and copy of archive

            if not isinstance(archive, List):
                raise type_error(archive, "archive", List)
            if not isinstance(archivec, List):
                raise type_error(archivec, "archivec", List)

            thelen: int = len(archive)
            if not isinstance(thelen, int):
                raise type_error(thelen, "len(archive)", int)
            if thelen != alen:
                raise ValueError(f"{alen} != len(archive)={len(archive)}?")

            thelen = len(archivec)
            if not isinstance(thelen, int):
                raise type_error(thelen, "len(archive)", int)
            if thelen != alen:
                raise ValueError(f"{alen} != len(archive)={len(archive)}?")

            # perform the pruning
            pruner.prune(archive, amax)

            thelen = len(archive)
            if not isinstance(thelen, int):
                raise type_error(thelen, "len(archive)", int)
            if thelen != alen:
                raise ValueError(
                    f"pruning messed up archive len: {alen} != "
                    f"len(archive)={len(archive)}?")

            # make sure that no element was deleted or added
            for ii, a in enumerate(archive):
                if not isinstance(a, MORecordY):
                    raise type_error(a, f"archive[{ii}]", MORecordY)
                if a.x != tag:
                    raise ValueError(f"a.x={a.x}!='{tag}'")
                if not isinstance(a.fs, np.ndarray):
                    raise type_error(a.fs, "a.fs", np.ndarray)
                for idx, b in enumerate(archivec):
                    if np.array_equal(a.fs, b.fs):
                        if a.x != b.x:
                            raise ValueError(f"a.fs={a.fs}==b.fs, "
                                             f"but a.x={a.x}!=b.x={b.x}")
                        if a.f != b.f:
                            raise ValueError(f"a.fs={a.fs}==b.fs, "
                                             f"but a.f={a.f}!=b.f={b.f}")
                        del archivec[idx]
                        break

    if nothing:
        raise ValueError("dimensions are empty!")
