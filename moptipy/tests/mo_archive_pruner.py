"""Functions for testing multi-objective archive pruners."""
from typing import Callable, Final, Iterable

import numpy as np
from numpy.random import Generator, default_rng

from moptipy.api.mo_archive import (
    MOArchivePruner,
    MORecord,
    check_mo_archive_pruner,
)
from moptipy.api.mo_problem import MOProblem
from moptipy.mock.mo_problem import MockMOProblem
from moptipy.tests.component import validate_component
from moptipy.utils.nputils import DEFAULT_NUMERICAL
from moptipy.utils.types import check_int_range, type_error


def __run_single_test(
        pruner_factory: Callable[[MOProblem], MOArchivePruner],
        random: Generator,
        dim: int,
        dt: np.dtype) -> bool:
    """
    Run a single test.

    :param pruner_factory: the factory for creating pruners
    :param random: the random number generator
    :param dim: the dimension
    :param dt: the data type
    :returns: `True` if a test was run, `False` if we need to try again
    """
    mop: Final[MockMOProblem] = MockMOProblem.for_dtype(dim, dt)
    if not isinstance(mop, MockMOProblem):
        raise type_error(mop, "new mock problem", MockMOProblem)

    tag: Final[str] = f"bla{random.integers(100)}"
    pruner = pruner_factory(mop)
    if not isinstance(pruner, MOArchivePruner):
        raise type_error(pruner, "pruner", MOArchivePruner)
    check_mo_archive_pruner(pruner)
    validate_component(pruner)

    alen = check_int_range(int(random.integers(2, 10)), "alen", 2, 9)
    amax = check_int_range(int(random.integers(1, alen + 1)), "amax", 1, alen)

    dim_mode = [False]
    if dim > 1:
        dim_mode.append(True)

    for use_collapse in dim_mode:
        archive_1: list[MORecord] = []
        archive_2: list[MORecord] = []
        archive_3: list[MORecord] = []

        collapse_dim: int = -1
        if use_collapse:
            collapse_dim = random.integers(dim)

        max_samples: int = 10000

        for _ in range(alen):
            needed: bool = True
            rec: MORecord | None = None
            fs: np.ndarray | None = None
            while needed:   # we make sure that all records are unique
                max_samples -= 1
                if max_samples <= 0:
                    return False
                fs = np.empty(dim, dt)
                if not isinstance(fs, np.ndarray):
                    raise type_error(fs, "fs", np.ndarray)
                if len(fs) != dim:
                    raise ValueError(f"len(fs)={len(fs)}!=dim={dim}")
                mop.sample(fs)
                rec = MORecord(tag, fs)
                if not isinstance(rec, MORecord):
                    raise type_error(rec, "rec", MORecord)
                needed = False
                for z in archive_1:
                    fs2 = z.fs
                    if np.array_equal(fs, fs2):
                        needed = True
                        break
            if (rec is None) or (fs is None):
                raise ValueError("huh?")
            archive_1.append(rec)
            rec2 = MORecord(tag, fs.copy())
            if (rec.x != rec2.x) \
                    or (not np.array_equal(rec2.fs, rec.fs)):
                raise ValueError(f"{rec} != {rec2}")
            archive_2.append(rec2)
            rec2 = MORecord(tag, fs.copy())
            archive_3.append(rec2)

        # done creating archive and copy of archive

        if not isinstance(archive_1, list):
            raise type_error(archive_1, "archive_1", list)
        if not isinstance(archive_2, list):
            raise type_error(archive_2, "archive_2", list)
        if not isinstance(archive_3, list):
            raise type_error(archive_3, "archive_3", list)

        thelen: int = len(archive_1)
        if not isinstance(thelen, int):
            raise type_error(thelen, "len(archive)", int)
        if thelen != alen:
            raise ValueError(
                f"{alen} != len(archive_1)={len(archive_1)}?")

        if use_collapse:
            collapse_value = archive_1[random.integers(alen)].fs[
                collapse_dim]
            for rec in archive_1:
                rec.fs[collapse_dim] = collapse_value
            for rec in archive_2:
                rec.fs[collapse_dim] = collapse_value
            for rec in archive_3:
                rec.fs[collapse_dim] = collapse_value

        # perform the pruning
        pruner.prune(archive_1, amax, len(archive_1))
        pruner.prune(archive_3, amax, len(archive_3))

        thelen = len(archive_1)
        if not isinstance(thelen, int):
            raise type_error(thelen, "len(archive_1)", int)
        if thelen != alen:
            raise ValueError(
                f"pruning messed up archive len: {alen} != "
                f"len(archive)={len(archive_1)}?")

        # make sure that no element was deleted or added
        for ii, a in enumerate(archive_1):
            if not isinstance(a, MORecord):
                raise type_error(a, f"archive[{ii}]", MORecord)
            for j in range(ii):
                if archive_1[j] is a:
                    raise ValueError(f"record {a} appears at "
                                     f"indexes {ii} and {j}!")
            if a.x != tag:
                raise ValueError(f"a.x={a.x}!={tag!r}")
            if not isinstance(a.fs, np.ndarray):
                raise type_error(a.fs, "a.fs", np.ndarray)
            b = archive_3[ii]
            if (a.x != b.x) or (not np.array_equal(a.fs, b.fs)):
                raise ValueError(
                    f"archive pruning not deterministic, archive_1[{ii}]={a}"
                    f" but archive_2[{ii}]={b}.")
            if not use_collapse:
                for idx, b in enumerate(archive_2):
                    if np.array_equal(a.fs, b.fs):
                        if a.x != b.x:
                            raise ValueError(
                                f"a.fs={a.fs}==b.fs, but "
                                f"a.x={a.x}!=b.x={b.x}")
                        del archive_2[idx]
                        break
    return True


def validate_mo_archive_pruner(
        pruner_factory: Callable[[MOProblem], MOArchivePruner],
        dimensions: Iterable[int],
        dtypes: Iterable[np.dtype] = DEFAULT_NUMERICAL) -> None:
    """
    Check whether an object is a moptipy multi-objective optimization pruner.

    This method checks whether the class is correct and whether the pruning
    follows the general contract: Interesting records in the list to be pruned
    are moved to the front, the ones to discard are moved to the back. No
    record is lost and none is duplicated.

    :param pruner_factory: the creator for the multi-objective archive pruner
        to test
    :param dimensions: the dimensions to simulate
    :param dtypes: the dtypes to test on
    :raises ValueError: if `mo_problem` is not a valid
        :class:`~moptipy.api.mo_problem.MOProblem`
    :raises TypeError: if values of the wrong types are encountered
    """
    if not callable(pruner_factory):
        raise type_error(pruner_factory, "pruner_factory", call=True)
    if not isinstance(dimensions, Iterable):
        raise type_error(dimensions, "dimensions", Iterable)
    if not isinstance(dtypes, Iterable):
        raise type_error(dtypes, "dtypes", Iterable)

    nothing: bool = True
    random: Final[Generator] = default_rng()
    for dim in dimensions:
        if not isinstance(dim, int):
            raise type_error(dim, "dimensions[i]", int)
        nothing = False
        for dt in dtypes:
            if not isinstance(dt, np.dtype):
                raise type_error(dt, "dt", np.dtype)
            while not __run_single_test(pruner_factory, random, dim, dt):
                pass

    if nothing:
        raise ValueError("dimensions are empty!")
